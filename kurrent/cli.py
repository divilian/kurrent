"""Command-line interface for kurrent.

Currently supported:

    kurrent ingest file.pdf
    kurrent ingest --local-metadata file.pdf
    kurrent ingest -r directoryOfPdfs
    kurrent ingest -y -r directoryOfPdfs
    kurrent search QUERY...
    kurrent search --metadata QUERY...
    kurrent search --text QUERY...
    kurrent search --semantic QUERY...

The default metadata mode is Crossref-enhanced metadata lookup. Use
--local-metadata to avoid network lookups.

The -y/--yes flag skips interactive metadata and heading review.

The default search mode is semantic chunk search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

from tqdm import tqdm

from kurrent.cli_display import (
    collapse_whitespace,
    context_window,
    distance_label,
    highlighted_metadata_value,
    pages_label,
    print_body,
    print_field,
    print_wrapped,
    reference_marker,
    section_label,
    separator_line,
    source_name_for_hit,
)
from kurrent.semantic_explainer import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    ChunkExplanation,
    SemanticExplanationBuffer,
)
from kurrent.semantic_highlighter import (
    semantically_highlighted_excerpt,
    semantically_highlighted_text,
)
from kurrent.terminal import QUIT_COMMANDS, is_quit_command

CROSSREF_REQUEST_INTERVAL_SECONDS = 1.0


class CliUsageError(Exception):
    """Raised for friendly CLI usage errors."""


def print_usage_error(message: str) -> None:
    """Print a friendly CLI usage error without a Python traceback."""

    print_wrapped(message, file=sys.stderr)


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


def review_section_headings(
    pdf_path: Path,
    use_llm_sectioning: bool,
) -> list[str] | None:
    """Let the user remove bogus rules-based section-heading candidates.

    When LLM-assisted sectioning is enabled, return None so the chunker can
    run the HeadingCandidate + Ollama pipeline and preserve candidate anchors.
    """

    if use_llm_sectioning:
        print()
        print("Section heading review")
        print("----------------------")
        print(
            "Using LLM-assisted section recognition during chunking. "
        )
        return None

    from kurrent.sectioner import detect_heading_candidates

    headings = detect_heading_candidates(pdf_path)
    print_heading_candidates(headings)

    if not headings:
        return []

    print()
    print("Enter comma-separated numbers to remove bogus headings.")
    print("Press Enter to keep all headings.")

    while True:
        raw = input("remove headings> ").strip()

        if is_quit_command(raw):
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


def accept_section_headings_without_review(
    pdf_path: Path,
    use_llm_sectioning: bool,
) -> list[str] | None:
    """Return headings accepted by -y/--yes, or None for LLM sectioning."""

    if use_llm_sectioning:
        print()
        print(
            "Using LLM-assisted section recognition during chunking. "
        )
        return None

    from kurrent.sectioner import detect_heading_candidates

    headings = detect_heading_candidates(pdf_path)
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


def already_ingested_outcome_if_complete(
    pdf_path: Path,
    store,
) -> IngestOutcome | None:
    """Return an existing ingest outcome if current chunks already exist.

    A document row alone is not enough to skip work, because a previous ingest
    may have failed after document registration but before chunk insertion.
    """

    from kurrent.chunker import chunker_version
    from kurrent.file_utils import sha256_file

    pdf_sha256 = sha256_file(pdf_path)
    existing = store.get_document_by_sha256(pdf_sha256)

    if existing is None:
        return None

    existing_chunks = store.get_chunks_for_document(
        doc_id=existing.doc_id,
        chunker_version=chunker_version(),
    )

    if not existing_chunks:
        return None

    return IngestOutcome(
        doc_id=existing.doc_id,
        already_existed=True,
    )


def ingest_pdf_with_metadata(
    pdf_path: Path,
    store,
    embedder,
    metadata,
    metadata_was_reviewed: bool,
    reviewed_headings: list[str] | None,
    use_llm_sectioning: bool,
    llm_progress_total_callback=None,
    llm_progress_callback=None,
) -> IngestOutcome:
    """Ingest one PDF using already-extracted metadata.

    This avoids doing Crossref lookup twice during interactive ingestion.
    reviewed_headings=None means the chunker should detect headings itself;
    a list means the CLI has supplied reviewed/accepted headings.

    use_llm_sectioning controls the automatic path when reviewed_headings is
    None.
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

    chunk_document(
        doc_id,
        store,
        reviewed_headings=reviewed_headings,
        use_llm_sectioning=use_llm_sectioning,
        llm_progress_total_callback=llm_progress_total_callback,
        llm_progress_callback=llm_progress_callback,
    )
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
    use_llm_sectioning: bool,
) -> IngestOutcome:
    """Ingest one PDF through the CLI workflow."""

    from kurrent.file_utils import normalize_path
    from kurrent.metadata_extractor import extract_metadata

    pdf_path = normalize_path(pdf_path)

    existing_outcome = already_ingested_outcome_if_complete(pdf_path, store)

    if existing_outcome is not None:
        print()
        print(f"({pdf_path.name} already ingested.)", flush=True)
        return existing_outcome

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
        reviewed_headings = accept_section_headings_without_review(
            pdf_path,
            use_llm_sectioning=use_llm_sectioning,
        )
    else:
        metadata = review_metadata(metadata)
        metadata_was_reviewed = True
        reviewed_headings = review_section_headings(
            pdf_path,
            use_llm_sectioning=use_llm_sectioning,
        )

    progress_bar = None

    def start_llm_progress(total: int) -> None:
        nonlocal progress_bar

        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None

        if total <= 0:
            print("No heading candidates will be sent to Ollama.", flush=True)
            return

        progress_bar = tqdm(
            total=total,
            desc="Ollama section candidates",
            unit="candidate",
        )

    def update_llm_progress(completed: int) -> None:
        if progress_bar is not None:
            progress_bar.update(completed)

    try:
        outcome = ingest_pdf_with_metadata(
            pdf_path=pdf_path,
            store=store,
            embedder=embedder,
            metadata=metadata,
            metadata_was_reviewed=metadata_was_reviewed,
            reviewed_headings=reviewed_headings,
            use_llm_sectioning=use_llm_sectioning,
            llm_progress_total_callback=(
                start_llm_progress
                if use_llm_sectioning and reviewed_headings is None
                else None
            ),
            llm_progress_callback=(
                update_llm_progress
                if use_llm_sectioning and reviewed_headings is None
                else None
            ),
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print()

    if outcome.already_existed:
        print(
            f"({pdf_path.name} already ingested.)",
            flush=True,
        )
    else:
        print("Created new document.", flush=True)

    return outcome


def ingest_targets(path: Path, recursive: bool) -> list[Path]:
    """Return PDF paths selected by CLI arguments."""

    from kurrent.file_utils import is_pdf, normalize_path

    path = normalize_path(path)

    if recursive:
        if path.is_file():
            raise CliUsageError(
                "Recursive ingest requires a directory. "
                f"Got a file instead: {path}"
            )

        if not path.exists():
            raise CliUsageError(
                "Recursive ingest requires a directory. "
                f"No such path exists: {path}"
            )

        if not path.is_dir():
            raise CliUsageError(
                "Recursive ingest requires a directory. "
                f"Got a non-directory path instead: {path}"
            )

        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".pdf"
        )

    if path.is_dir():
        raise CliUsageError(
            "Directory ingest requires -r/--recursive. "
            f"Got a directory: {path}"
        )

    if not path.exists():
        raise CliUsageError(f"No such PDF file: {path}")

    if not is_pdf(path):
        raise CliUsageError(
            "Ingest requires a PDF file. "
            f"Got a non-PDF path: {path}"
        )

    return [path]


def print_chunk_explanation(
    explanation: ChunkExplanation | None,
    waiting_message: str = "still thinking...",
) -> None:
    """Print a semantic relevance explanation, if available."""

    if explanation is None:
        print_field("why", waiting_message)
        return

    prefix = ""

    if explanation.relevant is False:
        prefix = "probably not relevant: "

    print_field("why", prefix + explanation.explanation)

    if explanation.error is not None:
        print_field("why error", explanation.error)


def document_for_hit(hit, state_store):
    """Return the parent document for a chunk hit, if available."""

    if state_store is None:
        return None

    try:
        return state_store.get_document(hit.doc_id)
    except Exception:
        return None


def print_document_summary(
    hit,
    index: int,
    total: int,
    search_text: str | None = None,
) -> None:
    """Print one document-level result summary."""

    title = highlighted_metadata_value(
        hit.title or "(untitled)",
        search_text,
    )
    authors = highlighted_metadata_value(
        hit.authors or "unknown author",
        search_text,
    )
    year = highlighted_metadata_value(
        hit.year if hit.year is not None else "n.d.",
        search_text,
    )

    print()
    print(separator_line())
    print_wrapped(f"Document {index}/{total}: {title}")
    print_field("authors", authors)
    print_field("year", year)

    if hit.score is not None:
        print_field("score", f"{hit.score:.4f}")


def print_document_detail(
    hit,
    index: int,
    total: int,
    search_text: str | None = None,
) -> None:
    """Print one document-level result in detail."""

    print()
    print_wrapped(f"Details for document {index}/{total}")
    print(separator_line())
    print_field(
        "title",
        highlighted_metadata_value(hit.title or "(untitled)", search_text),
    )
    print_field(
        "authors",
        highlighted_metadata_value(hit.authors or "unknown author", search_text),
    )
    print_field(
        "year",
        highlighted_metadata_value(
            hit.year if hit.year is not None else "n.d.",
            search_text,
        ),
    )

    if hit.score is not None:
        print_field("score", f"{hit.score:.4f}")


def chunk_excerpt(
    hit,
    search_text: str | None,
    semantic_query: str | None,
    embedder,
    max_chars: int,
) -> str:
    """Return an appropriately highlighted chunk excerpt."""

    if semantic_query is not None and embedder is not None:
        return semantically_highlighted_excerpt(
            hit.text,
            semantic_query,
            embedder,
            max_chars=max_chars,
        )

    return context_window(hit.text, search_text, width=max_chars)


def full_chunk_text(
    hit,
    search_text: str | None,
    semantic_query: str | None,
    embedder,
) -> str:
    """Return full chunk text with the appropriate highlighting."""

    if semantic_query is not None and embedder is not None:
        return semantically_highlighted_text(hit.text, semantic_query, embedder)

    return collapse_whitespace(hit.text)



def search_position_label(kind: str, index: int, total: int | None) -> str:
    """Return a result-position label, omitting total when it is uncertain."""

    if total is None:
        return f"{kind} {index}"

    return f"{kind} {index}/{total}"

def print_chunk_summary(
    hit,
    index: int,
    total: int | None,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
    show_distance: bool = False,
    state_store=None,
    explanation_buffer: SemanticExplanationBuffer | None = None,
    explanation: ChunkExplanation | None = None,
) -> None:
    """Print one chunk-level result summary."""

    document = document_for_hit(hit, state_store)
    title = (
        document.title
        if document is not None and document.title is not None
        else hit.title or source_name_for_hit(hit) or "(unknown document)"
    )

    print()
    print(separator_line())
    print_wrapped(
        f"{search_position_label('Chunk', index, total)}{reference_marker(hit)}"
    )
    print_field("title", title)

    if document is not None:
        print_field("authors", document.authors or "unknown author")
        print_field("year", document.year if document.year is not None else "n.d.")

    section = section_label(hit)
    if section is not None:
        print_field("section", section)

    if show_distance:
        print_field("distance", distance_label(hit))

    if explanation_buffer is not None:
        print_chunk_explanation(explanation)

    preview = chunk_excerpt(
        hit,
        search_text=search_text,
        semantic_query=semantic_query,
        embedder=embedder,
        max_chars=420,
    )
    print()
    print_body(preview, search_text=search_text)


def print_chunk_detail(
    hit,
    index: int,
    total: int | None,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
    show_distance: bool = False,
    explanation_buffer: SemanticExplanationBuffer | None = None,
) -> None:
    """Print one chunk-level result in detail."""

    print()
    print_wrapped(
        f"Details for {search_position_label('chunk', index, total)}"
        f"{reference_marker(hit)}"
    )
    print(separator_line())

    section = section_label(hit)
    if section is not None:
        print_field("section", section)

    pages = pages_label(hit)
    if pages is not None:
        print_field("pages", pages)

    source_name = source_name_for_hit(hit)
    if source_name is not None:
        print_field("source", source_name)

    if explanation_buffer is not None:
        explanation = explanation_buffer.get(hit, wait_seconds=20.0)
        print_chunk_explanation(
            explanation,
            waiting_message="still thinking...",
        )

    detail_text = full_chunk_text(
        hit,
        search_text=search_text,
        semantic_query=semantic_query,
        embedder=embedder,
    )
    print()
    print_body(detail_text, search_text=search_text)


def prompt_result_action() -> str:
    """Prompt for the next interactive search-result action."""

    try:
        return input("[Enter] next, d details, q quit > ").strip().lower()
    except EOFError:
        print()
        return "q"


def present_document_hits(
    hits,
    search_text: str | None = None,
) -> None:
    """Present document hits one at a time."""

    if not hits:
        print("No matching documents.")
        return

    total = len(hits)

    for i, hit in enumerate(hits, start=1):
        print_document_summary(
            hit,
            i,
            total,
            search_text=search_text,
        )

        while True:
            choice = prompt_result_action()

            if choice == "":
                break

            if choice == "d":
                print_document_detail(
                    hit,
                    i,
                    total,
                    search_text=search_text,
                )
                continue

            if is_quit_command(choice):
                return

            print("Please press Enter, or type d or q.")


def present_chunk_hits(
    hits,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
    show_distance: bool = False,
    state_store=None,
    explanation_buffer: SemanticExplanationBuffer | None = None,
) -> None:
    """Present chunk hits one at a time."""

    if not hits:
        print("No matching chunks.")
        return

    raw_total = len(hits)
    total_for_display = None if explanation_buffer is not None else raw_total
    displayed = 0
    skipped = 0

    for hit in hits:
        explanation = None

        if explanation_buffer is not None:
            explanation = explanation_buffer.get(hit, wait_seconds=8.0)

            if explanation is not None and explanation.relevant is False:
                skipped += 1
                continue

        displayed += 1
        print_chunk_summary(
            hit,
            displayed,
            total_for_display,
            search_text=search_text,
            semantic_query=semantic_query,
            embedder=embedder,
            show_distance=show_distance,
            state_store=state_store,
            explanation_buffer=explanation_buffer,
            explanation=explanation,
        )

        while True:
            choice = prompt_result_action()

            if choice == "":
                break

            if choice == "d":
                if explanation_buffer is not None:
                    refreshed = explanation_buffer.get(hit, wait_seconds=20.0)

                    if refreshed is not None and refreshed.relevant is False:
                        print_wrapped(
                            "This result was later judged not relevant, "
                            "so it is being skipped."
                        )
                        skipped += 1
                        break

                print_chunk_detail(
                    hit,
                    displayed,
                    total_for_display,
                    search_text=search_text,
                    semantic_query=semantic_query,
                    embedder=embedder,
                    show_distance=show_distance,
                    explanation_buffer=explanation_buffer,
                )
                continue

            if is_quit_command(choice):
                return

            print("Please press Enter, or type d or q.")

    if displayed == 0 and skipped:
        print("No chunks survived the Ollama relevance review.")

def run_search(args: argparse.Namespace) -> int:
    """Run the kurrent search command."""

    from kurrent.config import get_kurrent_state_paths
    from kurrent.searcher import Searcher
    from kurrent.state_store import StateStore

    query = " ".join(args.query).strip()

    if not query:
        raise CliUsageError("Search requires a non-empty query.")

    state_paths = get_kurrent_state_paths(args.state_dir)

    if not state_paths.sqlite_path.exists():
        raise CliUsageError(
            "No kurrent SQLite database exists yet. Ingest PDFs first, or pass "
            "--state-dir pointing to an existing kurrent state directory. "
            f"Expected database: {state_paths.sqlite_path}"
        )

    store = StateStore(state_paths.sqlite_path)

    try:
        if args.search_mode == "semantic":
            from kurrent.embedder import Embedder

            if not state_paths.chroma_path.exists():
                raise CliUsageError(
                    "No kurrent Chroma directory exists yet. Semantic search "
                    "requires embedded chunks. Ingest PDFs first, or pass "
                    "--state-dir pointing to an existing kurrent state directory. "
                    f"Expected Chroma directory: {state_paths.chroma_path}"
                )

            embedder = Embedder(chroma_path=state_paths.chroma_path)
            searcher = Searcher(state_store=store, embedder=embedder)
            hits = searcher.semantic_chunk_search(
                query,
                n_results=args.limit,
                max_distance=args.max_distance,
                include_reference_sections=args.include_reference_sections,
            )

            explanation_buffer = None

            if not args.no_explain:
                explanation_buffer = SemanticExplanationBuffer(
                    query=query,
                    hits=hits,
                    model=args.ollama_model,
                    ollama_url=args.ollama_url,
                    timeout_seconds=args.ollama_timeout,
                    max_workers=args.ollama_workers,
                )

            try:
                print_wrapped(f"Semantic search: {query!r}")
                print_wrapped(f"Hits: {len(hits)}")
                if explanation_buffer is not None:
                    print_wrapped(
                        "Explanations are being generated in the background. "
                        "Chunks judged not relevant will be skipped."
                    )
                present_chunk_hits(
                    hits,
                    semantic_query=query,
                    embedder=embedder,
                    show_distance=True,
                    state_store=store,
                    explanation_buffer=explanation_buffer,
                )
            finally:
                if explanation_buffer is not None:
                    explanation_buffer.close()

            return 0

        searcher = Searcher(state_store=store)

        if args.search_mode == "metadata":
            hits = searcher.metadata_search(query, limit=args.limit)
            print_wrapped(f"Metadata search: {query!r}")
            print_wrapped(f"Documents: {len(hits)}")
            present_document_hits(hits, search_text=query)
            return 0

        if args.search_mode == "text":
            hits = searcher.full_text_search(query, limit=args.limit)
            print_wrapped(f"Full-text search: {query!r}")
            print_wrapped(f"Chunks: {len(hits)}")
            present_chunk_hits(hits, search_text=query)
            return 0

        raise CliUsageError(f"Unknown search mode: {args.search_mode}")
    finally:
        store.close()

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

    try:
        pdf_paths = ingest_targets(args.path, recursive=args.recursive)
    except CliUsageError as exc:
        print()
        print_usage_error(str(exc))
        return 2

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
    print(
        "Sectioning mode:         "
        + (
            "rules-based"
            if args.rules_based_sections
            else "LLM-assisted"
        ),
        flush=True,
    )

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
            print("-" * 79)
            print(f"[{i}/{len(pdf_paths)}] {pdf_path}", flush=True)

            try:
                outcome = ingest_one_pdf(
                    pdf_path=pdf_path,
                    store=store,
                    embedder=embedder,
                    doi_lookup=doi_lookup,
                    crossref_mailto=crossref_mailto,
                    assume_yes=args.assume_yes,
                    use_llm_sectioning=not args.rules_based_sections,
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
                print_wrapped(f"Could not ingest {pdf_path}: {message}")
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
            print_wrapped(
                f"{result.pdf_path}: {result.error}",
                indent="  ",
                subsequent_indent="    ",
            )

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
    ingest_parser.add_argument(
        "--rules-based-sections",
        "--no-llm-sections",
        action="store_true",
        help=(
            "use the older rules-based section heading detector instead of "
            "LLM-assisted section recognition"
        ),
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

    search_parser = subparsers.add_parser(
        "search",
        help="search ingested kurrent documents",
    )
    search_mode_group = search_parser.add_mutually_exclusive_group()
    search_mode_group.add_argument(
        "--metadata",
        action="store_const",
        const="metadata",
        dest="search_mode",
        help="search title, authors, year, DOI, and PDF path.",
    )
    search_mode_group.add_argument(
        "--text",
        action="store_const",
        const="text",
        dest="search_mode",
        help="search stored chunk text with literal SQLite LIKE matching.",
    )
    search_mode_group.add_argument(
        "--semantic",
        action="store_const",
        const="semantic",
        dest="search_mode",
        help="search embedded chunks semantically (default).",
    )
    search_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="maximum number of search hits to print (default: 10).",
    )
    search_parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="semantic-search distance cutoff; lower is more similar.",
    )
    search_parser.add_argument(
        "--include-reference-sections",
        action="store_true",
        help="include reference/bibliography chunks in semantic results.",
    )
    search_parser.add_argument(
        "--no-explain",
        action="store_true",
        help="disable background Ollama explanations for semantic search hits.",
    )
    search_parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=(
            "Ollama model used for semantic-hit explanations "
            f"(default: {DEFAULT_OLLAMA_MODEL})."
        ),
    )
    search_parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL for explanations (default: {DEFAULT_OLLAMA_URL}).",
    )
    search_parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=45.0,
        help="seconds before one Ollama explanation request times out.",
    )
    search_parser.add_argument(
        "--ollama-workers",
        type=int,
        default=2,
        help="number of background Ollama explanation workers (default: 2).",
    )
    search_parser.add_argument(
        "query",
        nargs="+",
        help="search query text.",
    )
    search_parser.set_defaults(
        func=run_search,
        search_mode="semantic",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except CliUsageError as exc:
        print_usage_error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
