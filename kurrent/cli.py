"""Command-line interface for kurrent.

Currently supported:

    kurrent ingest file.pdf
    kurrent ingest --in-place file.pdf
    kurrent ingest --local-metadata file.pdf
    kurrent ingest -r directoryOfPdfs
    kurrent ingest -y -r directoryOfPdfs
    kurrent search QUERY...
    kurrent search --metadata QUERY...
    kurrent search --text QUERY...
    kurrent search --semantic QUERY...

The default metadata mode is Crossref-enhanced metadata lookup. Use
--local-metadata to avoid network lookups.

The default ingest storage mode copies PDFs into the kurrent state directory's
pdfs/ directory. Use --in-place to leave source PDFs where they are.

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
    ANSI_RESET,
    ansi_enabled,
    collapse_whitespace,
    context_window,
    distance_label,
    highlighted_metadata_value,
    pages_label,
    print_body,
    print_field,
    print_wrapped,
    reference_marker,
    terminal_width,
    section_label,
    separator_line,
    source_name_for_hit,
)
from kurrent.relevance_judge import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    RelevanceJudgment,
    RelevanceJudgmentBuffer,
)
from kurrent.semantic_highlighter import (
    semantically_highlighted_excerpt,
    semantically_highlighted_text,
)
from kurrent.terminal import QUIT_COMMANDS, is_quit_command
from kurrent.pdf_opener import open_pdf

CROSSREF_REQUEST_INTERVAL_SECONDS = 1.0
SEMANTIC_OVERFETCH_FACTOR = 5
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[93m"
ANSI_GRAY = "\033[90m"


class CliUsageError(Exception):
    """Raised for friendly CLI usage errors."""


def print_usage_error(message: str) -> None:
    """Print a friendly CLI usage error without a Python traceback."""

    print_wrapped(message, file=sys.stderr)


def colored_cli_text(text: str, ansi_code: str) -> str:
    """Return text wrapped in ANSI color when terminal output supports it."""

    if not ansi_enabled():
        return text

    return f"{ansi_code}{text}{ANSI_RESET}"


def red_prompt(text: str) -> str:
    """Return the main kurrent prompt colored red when possible."""

    return colored_cli_text(text, ANSI_RED)


def yellow_prompt(text: str) -> str:
    """Return the source-browser prompt colored yellow when possible."""

    return colored_cli_text(text, ANSI_YELLOW)


def yellow_menu_text(text: str) -> str:
    """Return source-browser menu text colored yellow when possible."""

    return colored_cli_text(text, ANSI_YELLOW)


def gray_status_text(text: str) -> str:
    """Return low-priority progress/status text colored gray when possible."""

    return colored_cli_text(text, ANSI_GRAY)


class StreamingWrappedPrinter:
    """Incrementally print streamed text with terminal-width wrapping.

    Ollama streaming chunks do not arrive on word boundaries, so this keeps the
    final partial word buffered until later text or finish() completes it.
    Whitespace is normalized similarly to print_wrapped(): paragraph newlines are
    preserved, while runs of spaces/tabs are collapsed to word separators.
    """

    def __init__(self, width: int | None = None) -> None:
        self.width = width if width is not None else terminal_width()
        self.width = max(1, self.width)
        self.line_len = 0
        self.buffer = ""

    def write(self, text: str) -> None:
        """Print a streamed text fragment, wrapping completed words."""

        if not text:
            return

        self.buffer += text.replace("\r", "")

        while self.buffer:
            if self.buffer[0] == "\n":
                print()
                self.line_len = 0
                self.buffer = self.buffer[1:]
                continue

            stripped = self.buffer.lstrip(" \t")

            if stripped != self.buffer:
                self.buffer = stripped
                continue

            next_break = self._next_break_index(self.buffer)

            if next_break is None:
                return

            word = self.buffer[:next_break]
            self.buffer = self.buffer[next_break:]
            self._write_word(word)

    def finish(self) -> None:
        """Flush any final buffered word after streaming completes."""

        final_word = self.buffer.strip()
        self.buffer = ""

        if final_word:
            self._write_word(final_word)

    @staticmethod
    def _next_break_index(text: str) -> int | None:
        for i, char in enumerate(text):
            if char in {" ", "\t", "\n"}:
                return i

        return None

    def _write_word(self, word: str) -> None:
        if not word:
            return

        word_len = len(word)

        if self.line_len > 0 and self.line_len + 1 + word_len > self.width:
            print()
            self.line_len = 0

        if self.line_len > 0:
            print(" ", end="", flush=True)
            self.line_len += 1

        print(word, end="", flush=True)
        self.line_len += word_len


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


@dataclass(slots=True)
class ExistingDocumentStatus:
    """State for a PDF whose content hash already exists in kurrent."""

    pdf_sha256: str
    document: object
    has_chunks: bool
    has_current_pipeline: bool



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


def existing_document_status(
    pdf_path: Path,
    store,
    use_llm_sectioning: bool = True,
) -> ExistingDocumentStatus | None:
    """Return existing-document status for this PDF content, if any."""

    from kurrent.chunker import chunker_version
    from kurrent.file_utils import sha256_file
    from kurrent.pipeline import current_text_pipeline_fingerprint

    pdf_sha256 = sha256_file(pdf_path)
    existing = store.get_document_by_sha256(pdf_sha256)

    if existing is None:
        return None

    existing_chunks = store.get_chunks_for_document(
        doc_id=existing.doc_id,
        chunker_version=chunker_version(),
    )
    pipeline_fingerprint = current_text_pipeline_fingerprint(
        use_llm_sectioning=use_llm_sectioning,
    )

    return ExistingDocumentStatus(
        pdf_sha256=pdf_sha256,
        document=existing,
        has_chunks=bool(existing_chunks),
        has_current_pipeline=(
            bool(existing_chunks)
            and store.document_has_current_pipeline(
                existing.doc_id,
                pipeline_fingerprint,
            )
        ),
    )


def print_already_ingested_message(
    source_pdf_path: Path,
    existing_document,
) -> None:
    """Explain that a PDF was skipped because its content already exists."""

    print()
    print(f"Already ingested: {source_pdf_path}", flush=True)

    title = existing_document.title or "(untitled)"
    print_wrapped(
        "This file has the same contents as an existing kurrent document:",
    )
    print_field("title", title)
    print_field("stored PDF", existing_document.pdf_path)
    print_wrapped("Skipping metadata, chunking, and embedding.")


def print_existing_document_needs_current_chunks_message(
    source_pdf_path: Path,
    existing_document,
) -> None:
    """Explain that an existing document needs current-version chunks."""

    print()
    print(f"Existing document: {source_pdf_path}", flush=True)
    print_wrapped(
        "Found existing document record for this PDF, but it has not been "
        "processed with the current extraction/sectioning/chunking pipeline.",
    )
    print_field("stored PDF", existing_document.pdf_path)
    print_wrapped(
        "Refreshing derived artifacts using the existing document record.",
    )


def ingest_pdf_with_metadata(
    pdf_path: Path,
    store,
    embedder,
    metadata,
    metadata_was_reviewed: bool,
    reviewed_headings: list[str] | None,
    use_llm_sectioning: bool,
    storage_mode: str,
    managed_pdf_dir: Path | None,
    llm_progress_total_callback=None,
    llm_progress_callback=None,
) -> IngestOutcome:
    """Ingest one PDF using already-extracted metadata.

    This avoids doing Crossref lookup twice during interactive ingestion.
    reviewed_headings=None means the chunker should detect headings itself;
    a list means the CLI has supplied reviewed/accepted headings.

    use_llm_sectioning controls the automatic path when reviewed_headings is
    None. storage_mode controls whether a new PDF is copied into kurrent's
    managed PDF directory or left in place.
    """

    from kurrent.chunker import chunk_document, chunker_version
    from kurrent.file_utils import is_pdf, normalize_path, sha256_file
    from kurrent.pipeline import current_text_pipeline_fingerprint
    from kurrent.schema import Document

    pdf_path = normalize_path(pdf_path)

    if not is_pdf(pdf_path):
        raise ValueError(f"No such PDF file {pdf_path}")

    pdf_sha256 = sha256_file(pdf_path)
    existing = store.get_document_by_sha256(pdf_sha256)
    already_existed = existing is not None

    if existing is None:
        stored_pdf_path = pdf_path

        if storage_mode == "managed":
            if managed_pdf_dir is None:
                raise ValueError("Managed ingest requires a managed PDF directory.")

            from kurrent.pdf_store import copy_pdf_to_managed_store

            stored_pdf_path = copy_pdf_to_managed_store(
                source_path=pdf_path,
                pdfs_dir=managed_pdf_dir,
                pdf_sha256=pdf_sha256,
            )

        document = Document.for_pdf(
            pdf_path=stored_pdf_path,
            pdf_sha256=pdf_sha256,
            storage_mode=storage_mode,
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

    pipeline_fingerprint = current_text_pipeline_fingerprint(
        reviewed_headings=reviewed_headings,
        use_llm_sectioning=use_llm_sectioning,
    )
    existing_current_version_chunks = store.get_chunks_for_document(
        doc_id=doc_id,
        chunker_version=chunker_version(),
    )
    stale_existing_chunks = (
        already_existed
        and bool(existing_current_version_chunks)
        and not store.document_has_current_pipeline(
            doc_id,
            pipeline_fingerprint,
        )
    )

    if stale_existing_chunks:
        embedder.delete_document(doc_id)

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
    storage_mode: str,
    managed_pdf_dir: Path | None,
) -> IngestOutcome:
    """Ingest one PDF through the CLI workflow."""

    from kurrent.file_utils import normalize_path
    from kurrent.metadata_extractor import extract_metadata

    pdf_path = normalize_path(pdf_path)

    existing_status = existing_document_status(
        pdf_path,
        store,
        use_llm_sectioning=use_llm_sectioning,
    )

    if existing_status is not None and existing_status.has_current_pipeline:
        print_already_ingested_message(
            source_pdf_path=pdf_path,
            existing_document=existing_status.document,
        )
        return IngestOutcome(
            doc_id=existing_status.document.doc_id,
            already_existed=True,
        )

    if existing_status is not None:
        print_existing_document_needs_current_chunks_message(
            source_pdf_path=pdf_path,
            existing_document=existing_status.document,
        )

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
            storage_mode=storage_mode,
            managed_pdf_dir=managed_pdf_dir,
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
        print("Updated existing document with current pipeline output.", flush=True)
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
    explanation: RelevanceJudgment | None,
    waiting_message: str = "checking relevance...",
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


def document_path_for_pipeline_message(document) -> Path | None:
    """Return the best available PDF path for a stale-pipeline hint."""

    path = getattr(document, "pdf_path", None)

    if path is None:
        path = getattr(document, "path", None)

    return path


def document_has_stale_search_pipeline(document, state_store) -> bool:
    """Return whether a search-result document has stale derived artifacts.

    Search is read-only, so stale detection must not trigger a refresh or make
    search fail. The stored document-level pipeline fingerprint is the source of
    truth here: chunk IDs and Chroma collection names can still look current
    when extractor or sectioner code has changed without a chunker-version bump.
    """

    if document is None or state_store is None:
        return False

    try:
        from kurrent.pipeline import is_current_text_pipeline_fingerprint

        pipeline_fingerprint = state_store.get_document_pipeline_fingerprint(
            document.doc_id,
        )
        return not is_current_text_pipeline_fingerprint(pipeline_fingerprint)
    except Exception:
        return False


def stale_pipeline_message(document) -> str:
    """Return a concise user-facing stale-pipeline refresh hint."""

    path = document_path_for_pipeline_message(document)

    if path is None:
        return "stale; run `kurrent ingest <pdf>` to refresh"

    return f"stale; run `kurrent ingest {path}` to refresh"


def print_stale_pipeline_warning(document, state_store) -> None:
    """Print a stale-pipeline warning for a search result, if needed."""

    if document_has_stale_search_pipeline(document, state_store):
        print_field("pipeline", stale_pipeline_message(document))




def all_documents_for_semantic_maintenance(store) -> list:
    """Return all documents known to kurrent for semantic-index checks.

    StateStore will likely grow a public list_documents() method. Until then,
    keep this helper tolerant of both real StateStore objects and lightweight
    unit-test fakes.
    """

    if hasattr(store, "list_documents"):
        return list(store.list_documents())

    if hasattr(store, "conn") and hasattr(store, "_row_to_document"):
        rows = store.conn.execute(
            """
            SELECT *
            FROM documents
            ORDER BY
                title IS NULL,
                title COLLATE NOCASE,
                pdf_path COLLATE NOCASE
            """
        ).fetchall()
        return [store._row_to_document(row) for row in rows]

    return []


def document_needs_semantic_refresh(document, store, embedder) -> bool:
    """Return whether a document needs refresh for current semantic search.

    A document needs semantic refresh if its stored text pipeline is stale, or if
    it has current text artifacts in SQLite but has not yet been indexed into the
    current semantic-search Chroma collection.
    """

    if document is None:
        return False

    try:
        from kurrent.pipeline import is_current_text_pipeline_fingerprint

        pipeline_fingerprint = store.get_document_pipeline_fingerprint(
            document.doc_id,
        )
        if not is_current_text_pipeline_fingerprint(pipeline_fingerprint):
            return True

        if hasattr(embedder, "has_document"):
            return not embedder.has_document(document.doc_id)
    except Exception:
        return False

    return False


def semantic_refresh_documents(store, embedder) -> list:
    """Return documents needing refresh before trustworthy semantic search."""

    return [
        document
        for document in all_documents_for_semantic_maintenance(store)
        if document_needs_semantic_refresh(document, store, embedder)
    ]


def prompt_refresh_semantic_documents(documents: list) -> bool:
    """Ask whether semantic search should refresh stale/missing documents."""

    if not documents:
        return False

    count = len(documents)
    noun = "document" if count == 1 else "documents"

    print()
    print_wrapped(
        f"{count} {noun} need refresh before semantic search is fully current."
    )
    print_wrapped(
        "They may have been ingested with an older extraction/sectioning/"
        "chunking pipeline, or may be missing from the current semantic index."
    )
    print_wrapped(
        "Refreshing now will rebuild derived artifacts as needed, update the "
        "current Chroma collection, and then run the search."
    )

    try:
        answer = input("Refresh them now? [Y/n] ").strip().lower()
    except EOFError:
        print()
        print_wrapped(
            "Continuing without refresh. Semantic search may omit stale or "
            "unindexed documents."
        )
        return False

    if answer in {"", "y", "yes"}:
        return True

    print_wrapped(
        "Continuing without refresh. Semantic search may omit stale or "
        "unindexed documents."
    )
    return False


def metadata_from_document(document):
    """Return ExtractedMetadata preserving existing document metadata."""

    from kurrent.schema import ExtractedMetadata

    return ExtractedMetadata(
        title=document.title,
        authors=document.authors,
        year=document.year,
        doi=document.doi,
    )


def refresh_documents_for_semantic_search(
    documents: list,
    store,
    embedder,
) -> tuple[int, int]:
    """Refresh documents before semantic search and return successes/failures."""

    if not documents:
        return 0, 0

    print()
    print_wrapped(f"Refreshing {len(documents)} documents for semantic search...")

    refreshed = 0
    failed = 0

    for i, document in enumerate(documents, start=1):
        print()
        print(f"[{i}/{len(documents)}] {document.pdf_path}", flush=True)

        try:
            ingest_pdf_with_metadata(
                pdf_path=document.pdf_path,
                store=store,
                embedder=embedder,
                metadata=metadata_from_document(document),
                metadata_was_reviewed=False,
                reviewed_headings=None,
                use_llm_sectioning=True,
                storage_mode=document.storage_mode,
                managed_pdf_dir=None,
            )
            refreshed += 1
        except Exception as exc:
            failed += 1
            message = f"{type(exc).__name__}: {exc}"
            print_wrapped(
                f"Could not refresh {document.pdf_path}: {message}",
            )

    print()
    print_wrapped(
        f"Semantic refresh complete: {refreshed} refreshed, {failed} failed."
    )

    if failed:
        print_wrapped(
            "Running search anyway. Results may omit documents that failed refresh."
        )
    else:
        print_wrapped("Semantic index is current. Running search.")

    return refreshed, failed


def offer_semantic_refresh_if_needed(store, embedder) -> None:
    """Offer to refresh stale/missing semantic-search documents before search."""

    documents = semantic_refresh_documents(store, embedder)

    if not documents:
        return

    if prompt_refresh_semantic_documents(documents):
        refresh_documents_for_semantic_search(
            documents=documents,
            store=store,
            embedder=embedder,
        )


def print_document_summary(
    hit,
    index: int,
    total: int,
    search_text: str | None = None,
    state_store=None,
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

    print_stale_pipeline_warning(hit, state_store)


def print_document_detail(
    hit,
    index: int,
    total: int,
    search_text: str | None = None,
    state_store=None,
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

    print_stale_pipeline_warning(hit, state_store)


def prompt_document_result_action() -> str:
    """Prompt for the next interactive document-result action."""

    try:
        return input("[Enter] next, d details, e edit metadata, q quit > ").strip().lower()
    except EOFError:
        print()
        return "q"


def open_pdf_result_message(result, purpose: str = "PDF") -> str:
    """Return a concise user-facing message for a PDF-open result."""

    if not result.success:
        return result.message or f"Could not open {purpose}: {result.path}"

    if result.page is not None and result.page_supported:
        return f"Opened {purpose} near p. {result.page}: {result.path}"

    if result.page is not None:
        return (
            f"Opened {purpose}: {result.path}. "
            f"Your viewer may not jump to p. {result.page} automatically."
        )

    return f"Opened {purpose}: {result.path}"


def open_pdf_for_metadata_edit(document) -> None:
    """Best-effort open of a document PDF in the user's default viewer."""

    path = document_path_for_pipeline_message(document)

    if path is None:
        print_wrapped("No PDF path is available for this document.")
        return

    print_wrapped("Opening PDF so you can inspect title/authors/year/DOI.")
    print_wrapped("When ready, return here and edit metadata.")

    result = open_pdf(path)
    print_wrapped(open_pdf_result_message(result, purpose="PDF"))


def document_hit_from_document(document, previous_hit):
    """Return a DocumentHit refreshed from an updated Document record."""

    from kurrent.schema import DocumentHit

    return DocumentHit(
        doc_id=document.doc_id,
        path=document.pdf_path,
        title=document.title,
        authors=document.authors,
        year=document.year,
        score=getattr(previous_hit, "score", None),
        best_chunk_id=getattr(previous_hit, "best_chunk_id", None),
    )


def metadata_changes(document, metadata) -> dict:
    """Return metadata update kwargs whose values differ from the document."""

    updates = metadata_update_kwargs(metadata)

    return {
        key: value
        for key, value in updates.items()
        if getattr(document, key) != value
    }


def edit_document_hit_metadata(hit, state_store):
    """Interactively edit metadata for a document search hit."""

    if state_store is None:
        print_wrapped("Metadata editing requires an open kurrent state store.")
        return hit

    document = state_store.get_document(hit.doc_id)

    if document is None:
        print_wrapped("Could not find this document in kurrent state.")
        return hit

    open_pdf_for_metadata_edit(document)
    edited_metadata = review_metadata(metadata_from_document(document))
    updates = metadata_changes(document, edited_metadata)

    if not updates:
        print_wrapped("Metadata unchanged.")
        return hit

    state_store.update_document_metadata(document.doc_id, **updates)
    refreshed = state_store.get_document(document.doc_id)

    if refreshed is None:
        print_wrapped("Metadata updated, but the refreshed document could not be loaded.")
        return hit

    print_wrapped("Metadata updated.")
    return document_hit_from_document(refreshed, hit)


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
    explanation_buffer: RelevanceJudgmentBuffer | None = None,
    explanation: RelevanceJudgment | None = None,
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

    print_stale_pipeline_warning(document, state_store)

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
    state_store=None,
    explanation_buffer: RelevanceJudgmentBuffer | None = None,
) -> None:
    """Print one chunk-level result in detail."""

    document = document_for_hit(hit, state_store)

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

    print_stale_pipeline_warning(document, state_store)

    if explanation_buffer is not None:
        explanation = explanation_buffer.get(hit, wait_seconds=0.0)
        print_chunk_explanation(
            explanation,
            waiting_message="checking relevance...",
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
    state_store=None,
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
            state_store=state_store,
        )

        while True:
            choice = prompt_document_result_action()

            if choice == "":
                break

            if choice == "d":
                print_document_detail(
                    hit,
                    i,
                    total,
                    search_text=search_text,
                    state_store=state_store,
                )
                continue

            if choice == "e":
                hit = edit_document_hit_metadata(hit, state_store)
                print_document_summary(
                    hit,
                    i,
                    total,
                    search_text=search_text,
                    state_store=state_store,
                )
                continue

            if is_quit_command(choice):
                return

            print("Please press Enter, or type d, e, or q.")


def present_chunk_hits(
    hits,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
    show_distance: bool = False,
    state_store=None,
    explanation_buffer: RelevanceJudgmentBuffer | None = None,
    max_display: int | None = None,
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
        if max_display is not None and displayed >= max_display:
            break

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
                print_chunk_detail(
                    hit,
                    displayed,
                    total_for_display,
                    search_text=search_text,
                    semantic_query=semantic_query,
                    embedder=embedder,
                    show_distance=show_distance,
                    state_store=state_store,
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

            offer_semantic_refresh_if_needed(store, embedder)

            candidate_limit = args.limit

            if not args.no_explain:
                candidate_limit = args.limit * SEMANTIC_OVERFETCH_FACTOR

            hits = searcher.semantic_chunk_search(
                query,
                n_results=candidate_limit,
                max_distance=args.max_distance,
                include_reference_sections=args.include_reference_sections,
            )

            explanation_buffer = None

            if not args.no_explain:
                explanation_buffer = RelevanceJudgmentBuffer(
                    query=query,
                    hits=hits,
                    model=args.ollama_model,
                    ollama_url=args.ollama_url,
                    timeout_seconds=args.ollama_timeout,
                    max_workers=args.ollama_workers,
                )

            try:
                print_wrapped(f"Semantic search: {query!r}")
                if explanation_buffer is None:
                    print_wrapped(f"Hits: {len(hits)}")
                else:
                    print_wrapped(f"Candidate chunks retrieved: {len(hits)}")
                    print_wrapped(f"Display limit: {args.limit}")
                    print_wrapped(
                        "Candidate chunks are being checked for relevance in the background. "
                        "Chunks judged not relevant will be skipped."
                    )
                present_chunk_hits(
                    hits,
                    semantic_query=query,
                    embedder=embedder,
                    show_distance=True,
                    state_store=store,
                    explanation_buffer=explanation_buffer,
                    max_display=args.limit if explanation_buffer is not None else None,
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
            present_document_hits(hits, search_text=query, state_store=store)
            return 0

        if args.search_mode == "text":
            hits = searcher.full_text_search(query, limit=args.limit)
            print_wrapped(f"Full-text search: {query!r}")
            print_wrapped(f"Chunks: {len(hits)}")
            present_chunk_hits(hits, search_text=query, state_store=store)
            return 0

        raise CliUsageError(f"Unknown search mode: {args.search_mode}")
    finally:
        store.close()



def print_converse_help() -> None:
    """Print available kurrent converse slash commands."""

    print()
    print("Available commands")
    print("------------------")
    print_wrapped("/help       Show this help.")
    print_wrapped("/sources    Open the source browser for the most recent answer.")
    print_wrapped("/open N     Open source N from the most recent answer.")
    print_wrapped("/quit       Leave converse.")


def _latest_converse_sources(turn):
    """Return source-navigation entries for the most recent converse turn."""

    if turn is None:
        return ()

    from kurrent.converser import evidence_sources

    return evidence_sources(turn.evidence)


def print_converse_sources(turn) -> None:
    """Print source-navigation entries for the most recent converse turn."""

    sources = _latest_converse_sources(turn)

    if not sources:
        print_wrapped("No sources are available yet. Ask a research question first.")
        return

    print()
    print_wrapped(yellow_menu_text("Sources from the most recent answer"))
    print_wrapped(yellow_menu_text("-----------------------------------"))

    for source in sources:
        print_wrapped(yellow_menu_text(f"{source.source_number}. {source.citation}"))


def open_converse_source(turn, source_number_text: str) -> None:
    """Open one source from the most recent converse turn."""

    sources = _latest_converse_sources(turn)

    if not sources:
        print_wrapped("No sources are available yet. Ask a research question first.")
        return

    try:
        source_number = int(source_number_text.strip())
    except ValueError:
        print_wrapped("Usage: /open N, where N is a source number from /sources.")
        return

    if not 1 <= source_number <= len(sources):
        print_wrapped(
            f"Source number out of range. Choose 1 through {len(sources)}."
        )
        return

    source = sources[source_number - 1]

    if source.pdf_path is None:
        print_wrapped(f"No PDF path is available for source {source_number}.")
        return

    result = open_pdf(source.pdf_path, page=source.page_start)
    print_wrapped(yellow_menu_text(open_pdf_result_message(result, purpose=f"source {source_number}")))


def prompt_source_action() -> str:
    """Prompt inside the converse source browser."""

    try:
        return input(yellow_prompt("sources> ")).strip()
    except EOFError:
        print()
        return "q"


def is_source_browser_quit(command: str) -> bool:
    """Return whether a source-browser command should return to kurrent>."""

    return command.strip().lower() in {
        "",
        "q",
        "/q",
        ":q",
        "quit",
        "/quit",
        "exit",
        "/exit",
    }


def browse_converse_sources(turn) -> None:
    """Open an interactive source browser for the most recent converse turn."""

    sources = _latest_converse_sources(turn)

    if not sources:
        print_wrapped("No sources are available yet. Ask a research question first.")
        return

    while True:
        print_converse_sources(turn)
        print_wrapped(yellow_menu_text("Type a source number to open it, or q to return to kurrent."))

        choice = prompt_source_action()

        if is_source_browser_quit(choice):
            return

        if choice.startswith("/open"):
            parts = choice.split(maxsplit=1)

            if len(parts) == 2:
                open_converse_source(turn, parts[1])
            else:
                print_wrapped(
                    "Type a source number to open it, or q to return to kurrent."
                )
            continue

        open_converse_source(turn, choice)


def handle_converse_command(command: str, last_turn) -> bool:
    """Handle a kurrent converse slash command.

    Return True when the caller should continue the session and False when it
    should exit.
    """

    command = command.strip()
    lowered = command.lower()

    if lowered in {"/quit", "/q", "/exit"}:
        return False

    if lowered == "/help":
        print_converse_help()
        return True

    if lowered == "/sources":
        browse_converse_sources(last_turn)
        return True

    if lowered.startswith("/open"):
        parts = command.split(maxsplit=1)

        if len(parts) != 2:
            print_wrapped("Usage: /open N, where N is a source number from /sources.")
            return True

        open_converse_source(last_turn, parts[1])
        return True

    print_wrapped(f"Unknown command: {command}")
    print_wrapped("Type /help for available commands.")
    return True

def run_converse(args: argparse.Namespace) -> int:
    """Run the kurrent converse command."""

    from kurrent.config import get_kurrent_state_paths
    from kurrent.converser import ConverseEngine, ConverseError
    from kurrent.embedder import Embedder
    from kurrent.searcher import Searcher
    from kurrent.state_store import StateStore

    state_paths = get_kurrent_state_paths(args.state_dir)

    if not state_paths.sqlite_path.exists():
        raise CliUsageError(
            "No kurrent SQLite database exists yet. Ingest PDFs first, or pass "
            "--state-dir pointing to an existing kurrent state directory. "
            f"Expected database: {state_paths.sqlite_path}"
        )

    if not state_paths.chroma_path.exists():
        raise CliUsageError(
            "No kurrent Chroma directory exists yet. Converse requires "
            "embedded chunks. Ingest PDFs first, or pass --state-dir pointing "
            "to an existing kurrent state directory. "
            f"Expected Chroma directory: {state_paths.chroma_path}"
        )

    store = StateStore(state_paths.sqlite_path)

    try:
        embedder = Embedder(chroma_path=state_paths.chroma_path)
        searcher = Searcher(state_store=store, embedder=embedder)

        offer_semantic_refresh_if_needed(store, embedder)

        engine = ConverseEngine(
            searcher=searcher,
            model=args.ollama_model,
            ollama_url=args.ollama_url,
            timeout_seconds=args.ollama_timeout,
            top_k=args.limit,
            max_distance=args.max_distance,
            include_reference_sections=args.include_reference_sections,
        )

        print()
        print_wrapped(red_prompt("Hi, what research question are you interested in today?"))

        def report_progress(message: str) -> None:
            print_wrapped(gray_status_text(f"  {message}"))

        last_turn = None

        while True:
            try:
                user_text = input(red_prompt("kurrent> ")).strip()
            except EOFError:
                print()
                return 0

            if is_quit_command(user_text):
                return 0

            if not user_text:
                continue

            if user_text.startswith("/"):
                if not handle_converse_command(user_text, last_turn):
                    return 0
                continue

            streamed_answer = False
            answer_printer = StreamingWrappedPrinter()

            def stream_answer_token(text: str) -> None:
                nonlocal streamed_answer

                if not streamed_answer:
                    print()
                    streamed_answer = True

                answer_printer.write(text)

            try:
                turn = engine.answer_user_turn(
                    user_text,
                    progress_callback=report_progress,
                    token_callback=stream_answer_token,
                )
            except ConverseError as exc:
                if streamed_answer:
                    answer_printer.finish()
                    print()
                print_wrapped(str(exc))
                continue

            last_turn = turn

            if streamed_answer:
                answer_printer.finish()
                print()
                print()
            else:
                print()
                print_wrapped(turn.assistant_text)
                print()

            browse_converse_sources(last_turn)
            print()
    finally:
        store.close()


def run_ingest(args: argparse.Namespace) -> int:
    """Run the kurrent ingest command."""

    print("Starting kurrent ingest...", flush=True)

    from kurrent.config import get_crossref_mailto, get_kurrent_state_paths

    state_paths = get_kurrent_state_paths(args.state_dir)
    storage_mode = "external" if args.in_place else "managed"

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

    if storage_mode == "managed":
        if state_paths.pdfs_path.exists():
            print(f"Managed PDF directory:   {state_paths.pdfs_path}", flush=True)
        else:
            print(
                "Managed PDF directory does not exist; it will be created: "
                f"{state_paths.pdfs_path}",
                flush=True,
            )

    print(
        "PDF storage mode:        "
        + ("managed" if storage_mode == "managed" else "in-place"),
    )
    print(f"Metadata mode:           {args.metadata_mode}", flush=True)
    print(
        "Sectioning mode:         "
        + (
            "rules-based"
            if args.rules_based_sections
            else "LLM-assisted"
        ),
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
                    storage_mode=storage_mode,
                    managed_pdf_dir=(
                        state_paths.pdfs_path
                        if storage_mode == "managed"
                        else None
                    ),
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
        "--in-place",
        "--external",
        action="store_true",
        dest="in_place",
        help=(
            "leave PDFs in their original locations instead of copying them "
            "into kurrent's managed pdfs directory."
        ),
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
        help="disable background Ollama relevance judging for semantic search hits.",
    )
    search_parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=(
            "Ollama model used for semantic-hit relevance judging "
            f"(default: {DEFAULT_OLLAMA_MODEL})."
        ),
    )
    search_parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL for relevance judging (default: {DEFAULT_OLLAMA_URL}).",
    )
    search_parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=45.0,
        help="seconds before one Ollama relevance judgment request times out.",
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


    converse_parser = subparsers.add_parser(
        "converse",
        help="open a stateful RAG research-inquiry session",
    )
    converse_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=8,
        help="number of semantic chunks to retrieve for each turn (default: 8).",
    )
    converse_parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="semantic-search distance cutoff; lower is more similar.",
    )
    converse_parser.add_argument(
        "--include-reference-sections",
        action="store_true",
        help="include reference/bibliography chunks in RAG evidence.",
    )
    converse_parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model used for RAG answers (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    converse_parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL for RAG answers (default: {DEFAULT_OLLAMA_URL}).",
    )
    converse_parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=120.0,
        help="seconds before one Ollama RAG answer request times out.",
    )
    converse_parser.set_defaults(func=run_converse)

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
