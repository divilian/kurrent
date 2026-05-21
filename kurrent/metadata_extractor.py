"""Extract lightweight bibliographic metadata from PDFs.

This module is intentionally conservative. PDF metadata is often missing or
misleading, so extraction proceeds through several cheap local signals:

    1. embedded PDF metadata
    2. DOI-like strings in early page text
    3. year-like strings in early page text
    4. title/author guesses from the first page
    5. filename fallback

The output is best-effort metadata for helping users recognize papers in
kurrent. It is not intended to be authoritative citation metadata.
"""

from __future__ import annotations

from pathlib import Path
import re

import pymupdf

from kurrent.schema import ExtractedMetadata


DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)

YEAR_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\b",
)

BAD_TITLE_PATTERNS = [
    "microsoft word",
    "untitled",
    "document",
    "manuscript",
    "final",
    "layout",
    "proof",
]


def clean_metadata_text(value: str | None) -> str | None:
    """Normalize a metadata string and turn empty values into None."""

    if value is None:
        return None

    value = " ".join(value.split()).strip()

    if not value:
        return None

    return value


def looks_like_bad_title(title: str | None) -> bool:
    """Return True when embedded PDF title metadata looks unhelpful."""

    title = clean_metadata_text(title)

    if title is None:
        return True

    lower_title = title.lower()

    if title.endswith(".doc") or title.endswith(".docx"):
        return True

    if title.endswith(".pdf"):
        return True

    return any(pattern in lower_title for pattern in BAD_TITLE_PATTERNS)


def extract_embedded_metadata(pdf_path: str | Path) -> ExtractedMetadata:
    """Extract title and author fields from embedded PDF metadata."""

    with pymupdf.open(pdf_path) as pdf:
        metadata = pdf.metadata or {}

    title = clean_metadata_text(metadata.get("title"))
    authors = clean_metadata_text(metadata.get("author"))

    if looks_like_bad_title(title):
        title = None

    return ExtractedMetadata(
        title=title,
        authors=authors,
    )


def extract_text_from_first_pages(
    pdf_path: str | Path,
    max_pages: int = 2,
) -> str:
    """Extract text from the first few pages of a PDF."""

    pieces: list[str] = []

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf[:max_pages]:
            pieces.append(page.get_text())

    return "\n".join(pieces)


def extract_doi(text: str) -> str | None:
    """Extract the first DOI-like string from text."""

    match = DOI_RE.search(text)

    if match is None:
        return None

    doi = match.group(0).rstrip(".,;:)]}'\"")
    return doi


PUBLICATION_YEAR_PATTERNS = [
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
               r"[a-z]*\s+(19\d{2}|20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(19\d{2}|20\d{2})\s+ACM\b", re.IGNORECASE),
    re.compile(r"©\s*(19\d{2}|20\d{2})\b", re.IGNORECASE),
]


def extract_year(text: str) -> int | None:
    """Extract a plausible publication year from text."""

    for pattern in PUBLICATION_YEAR_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1))

    years = [int(match.group(0)) for match in YEAR_RE.finditer(text)]
    plausible_years = [
        year
        for year in years
        if 1900 <= year <= 2100
    ]

    if not plausible_years:
        return None

    return max(plausible_years)


def first_nonempty_lines(text: str, limit: int = 20) -> list[str]:
    """Return the first non-empty lines from extracted text."""

    lines: list[str] = []

    for raw_line in text.splitlines():
        line = clean_metadata_text(raw_line)

        if line is None:
            continue

        lines.append(line)

        if len(lines) >= limit:
            break

    return lines


def looks_like_header_noise(line: str) -> bool:
    """Return True for common first-page lines unlikely to be the title."""

    lower_line = line.lower()

    if len(line) <= 3:
        return True

    if lower_line.startswith("http://") or lower_line.startswith("https://"):
        return True

    if "doi:" in lower_line:
        return True

    if lower_line.startswith("doi "):
        return True

    if lower_line.startswith("copyright"):
        return True

    if lower_line.startswith("©"):
        return True

    if lower_line in {"abstract", "introduction"}:
        return True

    if re.fullmatch(r"\d+", line):
        return True

    return False


def guess_title_from_first_page(text: str) -> str | None:
    """Guess a paper title from early first-page text.

    This heuristic picks the first substantial non-noise line. It deliberately
    avoids trying to be clever about multi-line titles for now.
    """

    for line in first_nonempty_lines(text):
        if looks_like_header_noise(line):
            continue

        if len(line) < 8:
            continue

        return line

    return None


def guess_authors_from_first_page(
    text: str,
    title: str | None,
) -> str | None:
    """Guess an author line from the first page.

    This uses the line after the guessed title when it looks author-like.
    It is intentionally conservative.
    """

    if title is None:
        return None

    lines = first_nonempty_lines(text)
    title_index: int | None = None

    for i, line in enumerate(lines):
        if line == title:
            title_index = i
            break

    if title_index is None:
        return None

    if title_index + 1 >= len(lines):
        return None

    candidate = lines[title_index + 1]

    if looks_like_header_noise(candidate):
        return None

    lower_candidate = candidate.lower()

    if "abstract" in lower_candidate:
        return None

    if "university" in lower_candidate:
        return None

    if "department" in lower_candidate:
        return None

    if len(candidate) > 200:
        return None

    return candidate


def guess_metadata_from_filename(pdf_path: str | Path) -> ExtractedMetadata:
    """Guess metadata from a PDF filename.

    This is a weak fallback for filenames such as:

        Epstein_2006_Generative_Social_Science.pdf
    """

    stem = Path(pdf_path).stem
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = " ".join(cleaned.split())

    year = extract_year(cleaned)
    title = cleaned

    if year is not None:
        title = re.sub(rf"\b{year}\b", "", title)
        title = " ".join(title.split())

    title = clean_metadata_text(title)

    return ExtractedMetadata(
        title=title,
        year=year,
    )


def merge_metadata(
    primary: ExtractedMetadata,
    fallback: ExtractedMetadata,
) -> ExtractedMetadata:
    """Fill missing metadata fields in primary from fallback."""

    return ExtractedMetadata(
        title=primary.title or fallback.title,
        authors=primary.authors or fallback.authors,
        year=primary.year or fallback.year,
        doi=primary.doi or fallback.doi,
    )


def extract_metadata(pdf_path: str | Path) -> ExtractedMetadata:
    """Extract best-effort bibliographic metadata for a PDF."""

    pdf_path = Path(pdf_path)

    embedded = extract_embedded_metadata(pdf_path)
    early_text = extract_text_from_first_pages(pdf_path, max_pages=2)

    doi = extract_doi(early_text)
    year = extract_year(early_text)

    first_page_title = guess_title_from_first_page(early_text)
    first_page_authors = guess_authors_from_first_page(
        early_text,
        first_page_title,
    )

    text_guess = ExtractedMetadata(
        title=first_page_title,
        authors=first_page_authors,
        year=year,
        doi=doi,
    )
    filename_guess = guess_metadata_from_filename(pdf_path)

    metadata = merge_metadata(embedded, text_guess)
    metadata = merge_metadata(metadata, filename_guess)

    return metadata


if __name__ == "__main__":

    # Smoke test / IPython playground.
    #
    # Run from IPython with:
    #
    #     run -m kurrent.metadata_extractor /path/to/paper.pdf
    #
    # Then inspect:
    #
    #     metadata
    #     pdf_path

    import sys

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path("/home/stephen/teaching/419/syllabus.pdf")

    metadata = extract_metadata(pdf_path)

    print(f"PDF:     {pdf_path}")
    print(f"Title:   {metadata.title}")
    print(f"Authors: {metadata.authors}")
    print(f"Year:    {metadata.year}")
    print(f"DOI:     {metadata.doi}")
