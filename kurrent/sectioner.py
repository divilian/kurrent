"""Detect and represent section structure in PDFs.

This module owns heading detection and conversion from reviewed headings to
SectionSpan objects. It deliberately contains no user-prompting code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import re

import pymupdf

from kurrent.file_utils import silence_mupdf_messages
from kurrent.schema import SectionSpan


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

REFERENCE_SECTION_TITLES = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
}


def normalize_section_title(section_title: str) -> str:
    """Normalize a section title for machine-facing classification."""

    normalized = section_title.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(".:")

    return normalized


def is_reference_section_title(section_title: str | None) -> bool:
    """Return whether a section title is a references/bibliography section.

    This intentionally does not classify "Related Work" or "Literature Review"
    as reference sections. Those are content sections, even though they cite a
    lot.
    """

    if section_title is None:
        return False

    return normalize_section_title(section_title) in REFERENCE_SECTION_TITLES


def looks_like_reference_text(text: str | None) -> bool:
    """Return whether text looks like bibliography/reference-list material.

    This intentionally looks for reference entries, not inline citations. A
    sentence like "underwater basket weaving[3] has been popular[15]" should
    not match the numbered-reference patterns, because the bracketed number is
    not at the beginning of a line or citation-entry-like unit.
    """

    if text is None:
        return False

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    collapsed = " ".join(text.split())

    # PDF extraction often collapses references into a single long line, so
    # inspect both real lines and citation-entry-like segments.
    units = list(raw_lines)
    units.extend(
        segment.strip()
        for segment in re.split(r"(?=\[\d+\]\s+)", collapsed)
        if segment.strip()
    )

    bracketed_reference_entries = 0
    numbered_reference_entries = 0

    for unit in units:
        if re.match(r"^\[\d+\]\s+\D", unit):
            bracketed_reference_entries += 1
        elif re.match(r"^\d+\.\s+[A-Z]", unit):
            numbered_reference_entries += 1

    if bracketed_reference_entries >= 3:
        return True

    if numbered_reference_entries >= 3:
        return True

    lower_text = collapsed.lower()

    reference_signals = [
        " journal of ",
        " proceedings of ",
        " proc. ",
        " physical review ",
        " nature ",
        " science ",
        " vol. ",
        " doi",
        " pp. ",
        " arxiv:",
        " springer",
        " elsevier",
        " cambridge university press",
        " oxford university press",
    ]

    signal_count = sum(
        1
        for signal in reference_signals
        if signal in f" {lower_text} "
    )

    # Require at least one entry-like marker before using journal/publisher
    # vocabulary as a fallback. This reduces false positives in ordinary prose.
    has_entry_marker = (
        bracketed_reference_entries > 0
        or numbered_reference_entries > 0
        or bool(re.search(r"\(\d{4}\)", collapsed))
    )

    return has_entry_marker and signal_count >= 3


def is_reference_section_chunk(chunk) -> bool:
    """Return whether a chunk-like object appears to be reference material."""

    if is_reference_section_title(getattr(chunk, "section_title", None)):
        return True

    return looks_like_reference_text(getattr(chunk, "text", None))



def normalize_line(line: str) -> str:
    """Normalize one extracted-text line for heading matching."""

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


def detect_heading_candidates(
    pdf_path: str | Path,
    max_pages: int = 8,
) -> list[str]:
    """Return plausible section-heading candidates from early PDF text."""

    silence_mupdf_messages()

    pdf_path = Path(pdf_path)
    candidates: list[str] = []

    with pymupdf.open(pdf_path) as pdf:
        pages_examined = min(len(pdf), max_pages)

        for page_index in range(pages_examined):
            page = pdf.load_page(page_index)
            text = page.get_text("text", sort=True) or ""

            for raw_line in text.splitlines():
                line = normalize_line(raw_line)

                if looks_like_heading(line):
                    candidates.append(line)

    candidates = dedupe_preserving_order(candidates)

    if len(candidates) < 2:
        return []

    return candidates


def parse_section_heading(
    heading: str,
) -> tuple[str | None, str | None]:
    """Split a visible section heading into number and title.

    Examples:
        "3.1 LLM Setup" -> ("3.1", "LLM Setup")
        "II. THE MODEL" -> ("II", "THE MODEL")
        "Abstract" -> (None, "Abstract")
    """

    heading = normalize_line(heading)

    patterns = [
        r"^(?P<number>\d+(?:\.\d+)*\.?)[ ]+(?P<title>.+)$",
        r"^(?P<number>[IVXLCDM]+)\.[ ]+(?P<title>.+)$",
        r"^(?P<number>[A-Z])\.[ ]+(?P<title>.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, heading)

        if match is None:
            continue

        number = match.group("number").rstrip(".")
        title = normalize_line(match.group("title"))

        return number or None, title or None

    return None, heading or None


def extract_pdf_lines_with_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract normalized nonempty text lines paired with 1-based page nums."""

    silence_mupdf_messages()

    lines: list[tuple[int, str]] = []

    with pymupdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True) or ""

            for raw_line in text.splitlines():
                line = normalize_line(raw_line)

                if line:
                    lines.append((page_num, line))

    return lines


def make_section_spans_from_headings(
    pdf_path: str | Path,
    doc_id: str,
    headings: Sequence[str],
) -> list[SectionSpan]:
    """Split PDF text into SectionSpan objects using accepted headings.

    Matching is currently line-based: if a normalized extracted line exactly
    matches one of the normalized headings, that line starts a new section.
    Text before the first accepted heading is kept as unsectioned front matter.

    If headings is empty, all extracted text becomes one unsectioned span.
    """

    accepted_by_normalized = {
        normalize_line(heading): heading
        for heading in headings
        if normalize_line(heading)
    }

    lines = extract_pdf_lines_with_pages(pdf_path)
    sections: list[SectionSpan] = []

    current_section_index: int | None = None
    current_section_number: str | None = None
    current_section_title: str | None = None
    current_start_page: int | None = None
    current_end_page: int | None = None
    current_lines: list[str] = []
    next_section_index = 0

    def emit_current() -> None:
        text = " ".join(current_lines).strip()

        if not text:
            return

        sections.append(
            SectionSpan(
                doc_id=doc_id,
                section_index=current_section_index,
                section_number=current_section_number,
                section_title=current_section_title,
                page_start=current_start_page,
                page_end=current_end_page,
                text=text,
            )
        )

    for page_num, line in lines:
        normalized = normalize_line(line)

        if normalized in accepted_by_normalized:
            emit_current()

            source_heading = accepted_by_normalized[normalized]
            section_number, section_title = parse_section_heading(
                source_heading,
            )

            current_section_index = next_section_index
            next_section_index += 1
            current_section_number = section_number
            current_section_title = section_title
            current_start_page = page_num
            current_end_page = page_num
            current_lines = [line]
            continue

        if not current_lines:
            # Front matter before the first accepted heading, or the whole
            # document if no headings were supplied.
            current_section_index = None
            current_section_number = None
            current_section_title = None
            current_start_page = page_num

        current_lines.append(line)
        current_end_page = page_num

    emit_current()

    return sections
