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
import math
import re
import shutil
import sys
import textwrap
import time

from tqdm import tqdm


QUIT_COMMANDS = {":q", ":quit", "done", "quit", "exit"}
CROSSREF_REQUEST_INTERVAL_SECONDS = 1.0


class CliUsageError(Exception):
    """Raised for friendly CLI usage errors."""


def print_wrapped(
    text: str,
    indent: str = "",
    subsequent_indent: str | None = None,
    width: int | None = None,
    file=None,
) -> None:
    """Print user-facing CLI prose wrapped to the terminal width."""

    if width is None:
        width = shutil.get_terminal_size(fallback=(79, 20)).columns

    if subsequent_indent is None:
        subsequent_indent = indent

    print(
        textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        file=file,
    )


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
        print(f"Created new kurrent ID: {outcome.doc_id}", flush=True)

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


ANSI_BOLD = "\033[1m"
ANSI_BOLD_YELLOW = "\033[1;33m"
ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

SEMANTIC_HIGHLIGHT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "might", "more",
    "most", "no", "not", "of", "on", "or", "our", "out", "over", "she",
    "should", "so", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "under",
    "up", "was", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your",
}

SEMANTIC_HIGHLIGHT_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'-]{2,}\b")


def ansi_enabled() -> bool:
    """Return whether ANSI formatting should be used for terminal output."""

    if sys.stdout is None:
        return False

    if not sys.stdout.isatty():
        return False

    if "NO_COLOR" in __import__("os").environ:
        return False

    if __import__("os").environ.get("TERM") == "dumb":
        return False

    return True


def terminal_width() -> int:
    """Return the current terminal width, with a conservative fallback."""

    return shutil.get_terminal_size(fallback=(79, 20)).columns


def visible_len(text: str) -> int:
    """Return display width after ignoring ANSI color codes."""

    return len(ANSI_ESCAPE_RE.sub("", str(text)))


def wrapped_lines(
    text: str,
    indent: str = "",
    subsequent_indent: str | None = None,
    width: int | None = None,
) -> list[str]:
    """Return terminal-width-wrapped lines, counting ANSI escapes as zero."""

    if width is None:
        width = terminal_width()

    if subsequent_indent is None:
        subsequent_indent = indent

    output: list[str] = []

    for raw_line in str(text).splitlines() or [""]:
        words = raw_line.split()

        if not words:
            output.append(indent.rstrip())
            continue

        prefix = indent
        available = max(1, width - visible_len(prefix))
        current = ""
        current_len = 0

        for word in words:
            word_len = visible_len(word)

            if not current:
                current = word
                current_len = word_len
                continue

            if current_len + 1 + word_len <= available:
                current += " " + word
                current_len += 1 + word_len
                continue

            output.append(prefix + current)
            prefix = subsequent_indent
            available = max(1, width - visible_len(prefix))
            current = word
            current_len = word_len

        if current:
            output.append(prefix + current)

    return output


def print_wrapped(
    text: str,
    indent: str = "",
    subsequent_indent: str | None = None,
    width: int | None = None,
    file=None,
) -> None:
    """Print user-facing CLI prose wrapped to the terminal width."""

    for line in wrapped_lines(
        text,
        indent=indent,
        subsequent_indent=subsequent_indent,
        width=width,
    ):
        print(line, file=file)


def separator_line() -> str:
    """Return a separator line that fits the current terminal width."""

    return "-" * min(terminal_width(), 79)


def collapse_whitespace(text: str) -> str:
    """Normalize text to a single display-friendly line."""

    return " ".join(text.split())


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


def context_window(
    text: str,
    search_text: str | None,
    width: int = 240,
) -> str:
    """Return a display window centered around the first literal match."""

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


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return cosine similarity for two embedding vectors."""

    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def semantic_windows(text: str, words_per_window: int = 70) -> list[str]:
    """Return overlapping word windows for choosing a semantic excerpt."""

    words = collapse_whitespace(text).split()

    if not words:
        return []

    if len(words) <= words_per_window:
        return [" ".join(words)]

    stride = max(1, words_per_window // 2)
    windows = []

    for start in range(0, len(words), stride):
        window_words = words[start:start + words_per_window]

        if len(window_words) < 12 and windows:
            break

        windows.append(" ".join(window_words))

        if start + words_per_window >= len(words):
            break

    return windows


def best_semantic_excerpt(
    text: str,
    query: str,
    embedder,
    max_chars: int,
) -> str:
    """Return the chunk excerpt whose local window best matches the query."""

    collapsed = collapse_whitespace(text)

    if len(collapsed) <= max_chars:
        return collapsed

    windows = semantic_windows(collapsed)

    if not windows:
        return context_window(collapsed, None, width=max_chars)

    embeddings = embedder.generate_embeddings([query] + windows)
    query_embedding = embeddings[0]
    window_embeddings = embeddings[1:]

    best_index = max(
        range(len(windows)),
        key=lambda i: cosine_similarity(query_embedding, window_embeddings[i]),
    )
    best_window = windows[best_index]
    best_start = collapsed.find(best_window)

    if best_start < 0:
        return context_window(collapsed, None, width=max_chars)

    best_center = best_start + len(best_window) // 2
    start = max(0, best_center - max_chars // 2)
    end = min(len(collapsed), start + max_chars)
    start = max(0, end - max_chars)

    excerpt = collapsed[start:end].strip()

    if start > 0:
        excerpt = "[...] " + excerpt

    if end < len(collapsed):
        excerpt = excerpt + " [...]"

    return excerpt


def semantic_candidate_words(text: str) -> list[str]:
    """Return unique content words eligible for semantic highlighting."""

    words: list[str] = []
    seen: set[str] = set()

    for match in SEMANTIC_HIGHLIGHT_TOKEN_RE.finditer(text):
        word = match.group(0)
        key = word.lower().strip("'-")

        if len(key) < 4:
            continue

        if key in SEMANTIC_HIGHLIGHT_STOPWORDS:
            continue

        if key in seen:
            continue

        seen.add(key)
        words.append(word)

    return words


def semantic_highlight_tiers(
    text: str,
    query: str,
    embedder,
) -> dict[str, str]:
    """Assign candidate words to bold/yellow/red semantic-highlight tiers."""

    candidates = semantic_candidate_words(text)

    if not candidates:
        return {}

    embeddings = embedder.generate_embeddings([query] + candidates)
    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    scored = []

    for word, embedding in zip(candidates, candidate_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scored.append((word.lower().strip("'-"), score))

    scored.sort(key=lambda item: item[1], reverse=True)

    if not scored or scored[0][1] < 0.12:
        return {}

    highlight_count = min(18, max(5, math.ceil(len(scored) * 0.18)))
    highlighted = scored[:highlight_count]

    red_count = max(1, math.ceil(len(highlighted) * 0.15))
    yellow_count = max(1, math.ceil(len(highlighted) * 0.30))

    tiers: dict[str, str] = {}

    for i, (word, score) in enumerate(highlighted):
        if score < 0.12:
            continue

        if i < red_count:
            tiers[word] = "red"
        elif i < red_count + yellow_count:
            tiers[word] = "yellow"
        else:
            tiers[word] = "bold"

    return tiers


def apply_semantic_highlights(text: str, tiers: dict[str, str]) -> str:
    """Apply semantic-highlight tiers to matching words in display text."""

    if not ansi_enabled() or not tiers:
        return text

    def replace(match: re.Match) -> str:
        word = match.group(0)
        key = word.lower().strip("'-")
        tier = tiers.get(key)

        if tier == "red":
            return f"{ANSI_BOLD_RED}{word}{ANSI_RESET}"

        if tier == "yellow":
            return f"{ANSI_BOLD_YELLOW}{word}{ANSI_RESET}"

        if tier == "bold":
            return f"{ANSI_BOLD}{word}{ANSI_RESET}"

        return word

    return SEMANTIC_HIGHLIGHT_TOKEN_RE.sub(replace, text)


def semantically_highlighted_excerpt(
    text: str,
    query: str,
    embedder,
    max_chars: int,
) -> str:
    """Return a semantic excerpt with three-tier semantic word highlighting."""

    excerpt = best_semantic_excerpt(
        text,
        query,
        embedder,
        max_chars=max_chars,
    )
    tiers = semantic_highlight_tiers(excerpt, query, embedder)
    return apply_semantic_highlights(excerpt, tiers)


def semantically_highlighted_text(text: str, query: str, embedder) -> str:
    """Return full text with semantic word highlighting applied."""

    collapsed = collapse_whitespace(text)
    tiers = semantic_highlight_tiers(collapsed, query, embedder)
    return apply_semantic_highlights(collapsed, tiers)


def print_field(label: str, value: object | None) -> None:
    """Print a wrapped label/value line for search results."""

    if value is None:
        return

    label_text = f"{label}:"
    indent = "  "
    subsequent = " " * (len(indent) + len(label_text) + 1)
    print_wrapped(
        f"{label_text} {value}",
        indent=indent,
        subsequent_indent=subsequent,
    )


def print_body(text: str, search_text: str | None = None) -> None:
    """Print wrapped body text for a result preview or detail view."""

    if search_text is not None:
        text = bold_matches(text, search_text)

    print_wrapped(text, indent="  ", subsequent_indent="  ")


def section_label(hit) -> str | None:
    """Return a compact section label for a chunk hit, if available."""

    pieces = []

    if hit.section_number is not None:
        pieces.append(str(hit.section_number))

    if hit.section_title is not None:
        pieces.append(hit.section_title)

    if not pieces:
        return None

    return " ".join(pieces)


def reference_marker(hit) -> str:
    """Return a visible marker for reference-section hits."""

    from kurrent.sectioner import is_reference_section_chunk

    if is_reference_section_chunk(hit):
        return " [REFERENCE SECTION]"

    return ""


def source_name_for_hit(hit) -> str | None:
    """Return a display-friendly source filename without exposing full paths."""

    if hit.path is None:
        return None

    return hit.path.name


def pages_label(hit) -> str | None:
    """Return a compact page-range label, if page data is available."""

    if hit.page_start is None and hit.page_end is None:
        return None

    if hit.page_start == hit.page_end:
        return f"p. {hit.page_start}"

    return f"pp. {hit.page_start}–{hit.page_end}"


def distance_label(hit) -> str | None:
    """Return a formatted semantic distance, if present."""

    if hit.distance is None:
        return None

    return f"{hit.distance:.4f}"


def document_for_hit(hit, store):
    """Return the parent document for a chunk hit, if available."""

    try:
        return store.get_document(hit.doc_id)
    except Exception:
        return None


def print_document_summary(hit, index: int, total: int) -> None:
    """Print a document-level search result summary."""

    print()
    print(f"Document {index} of {total}")
    print(separator_line())
    print_field("Title", hit.title or "(untitled)")
    print_field("Authors", hit.authors or "unknown author")
    print_field("Year", hit.year if hit.year is not None else "n.d.")


def print_document_detail(hit, index: int, total: int) -> None:
    """Print document-level search result details without internal IDs."""

    print()
    print(f"Document {index} of {total}: details")
    print(separator_line())
    print_field("Title", hit.title or "(untitled)")
    print_field("Authors", hit.authors or "unknown author")
    print_field("Year", hit.year if hit.year is not None else "n.d.")
    if hit.path is not None:
        print_field("Source", hit.path.name)


def print_chunk_summary(
    hit,
    index: int,
    total: int,
    store,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
    show_distance: bool = False,
) -> None:
    """Print a chunk-level search result summary without internal IDs."""

    document = document_for_hit(hit, store)

    print()
    print(f"Chunk {index} of {total}{reference_marker(hit)}")
    print(separator_line())
    print_field("Title", (document.title if document else hit.title) or "(untitled)")
    print_field("Authors", document.authors if document else None)
    print_field("Year", document.year if document else None)
    print_field("Section", section_label(hit))

    if show_distance:
        print_field("Distance", distance_label(hit))

    print()

    if semantic_query is not None and embedder is not None:
        text = semantically_highlighted_excerpt(
            hit.text,
            semantic_query,
            embedder,
            max_chars=520,
        )
    else:
        text = context_window(hit.text, search_text, width=520)

    print_body(text, search_text=search_text)


def print_chunk_detail(
    hit,
    index: int,
    total: int,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
) -> None:
    """Print chunk-level search result details without internal IDs."""

    print()
    print(f"Chunk {index} of {total}: details{reference_marker(hit)}")
    print(separator_line())
    print_field("Section", section_label(hit))
    print_field("Pages", pages_label(hit))
    print_field("Source", source_name_for_hit(hit))
    print()

    if semantic_query is not None and embedder is not None:
        text = semantically_highlighted_text(hit.text, semantic_query, embedder)
        print_body(text)
    else:
        text = collapse_whitespace(hit.text)
        print_body(text, search_text=search_text)


def prompt_result_action(has_next: bool) -> str:
    """Prompt for result navigation and return enter/d/q."""

    if has_next:
        prompt = "[Enter] next, d details, q quit > "
    else:
        prompt = "[Enter] finish, d details, q quit > "

    while True:
        try:
            response = input(prompt).strip().lower()
        except EOFError:
            print()
            return "q"

        if response == "":
            return "enter"

        if response in {"d", "q"} | QUIT_COMMANDS:
            return "q" if response in QUIT_COMMANDS else response

        print("Please press Enter, type d for details, or type q to quit.")


def page_document_hits(hits) -> None:
    """Present document hits one at a time."""

    if not hits:
        print("No matching documents.")
        return

    total = len(hits)

    for i, hit in enumerate(hits, start=1):
        print_document_summary(hit, i, total)
        action = prompt_result_action(has_next=i < total)

        if action == "q":
            return

        if action == "d":
            print_document_detail(hit, i, total)
            action = prompt_result_action(has_next=i < total)

            if action == "q":
                return


def page_chunk_hits(
    hits,
    store,
    search_text: str | None = None,
    semantic_query: str | None = None,
    embedder=None,
    show_distance: bool = False,
) -> None:
    """Present chunk hits one at a time."""

    if not hits:
        print("No matching chunks.")
        return

    total = len(hits)

    for i, hit in enumerate(hits, start=1):
        print_chunk_summary(
            hit,
            i,
            total,
            store=store,
            search_text=search_text,
            semantic_query=semantic_query,
            embedder=embedder,
            show_distance=show_distance,
        )
        action = prompt_result_action(has_next=i < total)

        if action == "q":
            return

        if action == "d":
            print_chunk_detail(
                hit,
                i,
                total,
                search_text=search_text,
                semantic_query=semantic_query,
                embedder=embedder,
            )
            action = prompt_result_action(has_next=i < total)

            if action == "q":
                return


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

            print_wrapped(f"Semantic search: {query!r}")
            print_wrapped(f"Hits: {len(hits)}")
            page_chunk_hits(
                hits,
                store=store,
                semantic_query=query,
                embedder=embedder,
                show_distance=True,
            )
            return 0

        searcher = Searcher(state_store=store)

        if args.search_mode == "metadata":
            hits = searcher.metadata_search(query, limit=args.limit)
            print_wrapped(f"Metadata search: {query!r}")
            print_wrapped(f"Documents: {len(hits)}")
            page_document_hits(hits)
            return 0

        if args.search_mode == "text":
            hits = searcher.full_text_search(query, limit=args.limit)
            print_wrapped(f"Full-text search: {query!r}")
            print_wrapped(f"Chunks: {len(hits)}")
            page_chunk_hits(hits, store=store, search_text=query)
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
