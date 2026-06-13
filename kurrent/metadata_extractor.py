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
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import pymupdf

from kurrent.file_utils import silence_mupdf_messages
from kurrent.pdf_text_extractor import sanitize_extracted_text
from kurrent.metadata_cleaning import (
    clean_author_metadata_text,
    clean_metadata_text,
    clean_title_metadata_text,
)
from kurrent.schema import ExtractedMetadata

__all__ = [
    "clean_author_metadata_text",
    "clean_metadata_text",
    "looks_like_bad_title",
    "extract_embedded_metadata",
    "extract_text_from_first_pages",
    "extract_doi",
    "extract_year",
    "first_nonempty_lines",
    "looks_like_header_noise",
    "guess_title_from_first_page",
    "guess_authors_from_first_page",
    "guess_metadata_from_filename",
    "metadata_from_crossref_work",
    "lookup_crossref_metadata",
    "merge_metadata",
    "extract_metadata",
]

silence_mupdf_messages()


DOI_URL_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/(?P<doi>10\.\d{4,9}/[^\s<>'\"]+)",
    re.IGNORECASE,
)

DOI_RE = re.compile(
    r"\b(?P<doi>10\.\d{4,9}/[^\s<>'\"]+)",
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

CROSSREF_API_BASE_URL = "https://api.crossref.org/works"


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

    title = clean_title_metadata_text(metadata.get("title"))
    authors = clean_author_metadata_text(metadata.get("author"))

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
            pieces.append(sanitize_extracted_text(page.get_text()))

    return sanitize_extracted_text("\n".join(pieces))


def _clean_doi_candidate(candidate: str) -> str | None:
    """Clean and validate one DOI candidate extracted from PDF text."""

    candidate = candidate.strip()
    candidate = candidate.replace("\u200b", "")
    candidate = candidate.replace("\ufeff", "")

    candidate = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        candidate,
        flags=re.IGNORECASE,
    )

    # PNAS supplement URLs often look like:
    #
    #     10.1073/pnas.2400689121/-/DCSupplemental
    #
    # The DOI is the part before /-/.
    candidate = candidate.split("/-/")[0]

    # Drop common trailing punctuation from surrounding prose.
    candidate = candidate.rstrip(".,;:)]}'\"")

    if not candidate.lower().startswith("10."):
        return None

    if "/" not in candidate:
        return None

    prefix, suffix = candidate.split("/", maxsplit=1)

    if not prefix or not suffix:
        return None

    # Very short / non-specific suffixes like "pnas" are usually fragments
    # produced by line-wrapped DOI URLs, not usable DOIs.
    if not any(char.isdigit() for char in suffix):
        return None

    return candidate


def _doi_candidate_score(candidate: str, from_doi_url: bool) -> tuple[int, int]:
    """Score DOI candidates so complete DOI URLs beat short fragments."""

    return (int(from_doi_url), len(candidate))


def extract_doi(text: str) -> str | None:
    """Extract the best DOI-like string from text.

    Prefer DOI candidates from complete DOI URLs. Otherwise prefer the
    longest cleaned DOI-like candidate. This avoids choosing early fragments
    such as ``10.1073/pnas`` when a later footer contains the full DOI.
    """

    candidates: list[tuple[tuple[int, int], str]] = []

    for match in DOI_URL_RE.finditer(text):
        doi = _clean_doi_candidate(match.group("doi"))

        if doi is not None:
            candidates.append((_doi_candidate_score(doi, True), doi))

    for match in DOI_RE.finditer(text):
        doi = _clean_doi_candidate(match.group("doi"))

        if doi is not None:
            candidates.append((_doi_candidate_score(doi, False), doi))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


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


TITLE_CONTINUATION_END_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
}

AFFILIATION_MARKERS = {
    "college",
    "department",
    "faculty",
    "institute",
    "laboratory",
    "school",
    "university",
}


def _metadata_line_key(line: str) -> str:
    """Normalize one line for comparing extracted metadata text."""

    return re.sub(r"\s+", " ", line).strip().lower()


def _last_word(line: str) -> str | None:
    """Return the last word-like token in a metadata line."""

    words = re.findall(r"[A-Za-z]+", line)

    if not words:
        return None

    return words[-1].lower()


def _looks_like_affiliation_line(line: str) -> bool:
    """Return True when a line looks like institution/affiliation text."""

    lower_line = line.lower()

    if "@" in line:
        return True

    return any(marker in lower_line for marker in AFFILIATION_MARKERS)


def _looks_like_title_continuation(previous_line: str, candidate: str) -> bool:
    """Return True when candidate likely continues a wrapped title line.

    The deliberately conservative rule here fixes common wrapped titles such
    as ``Natural-Language Multi-Agent Simulations of`` followed by
    ``Argumentative Opinion Dynamics`` without turning ordinary journal/section
    headings into accidental title prefixes.
    """

    if looks_like_header_noise(candidate):
        return False

    if _looks_like_affiliation_line(candidate):
        return False

    if len(candidate) < 4 or len(candidate) > 160:
        return False

    previous_last_word = _last_word(previous_line)

    if previous_last_word in TITLE_CONTINUATION_END_WORDS:
        return True

    # Hyphenated line wrapping can split title words across lines.
    if previous_line.endswith("-"):
        return True

    return False


def _find_title_line_span(lines: list[str], title: str) -> tuple[int, int] | None:
    """Find the inclusive-exclusive line span that produced a title guess."""

    title_key = _metadata_line_key(title)

    for start in range(len(lines)):
        pieces: list[str] = []

        for end in range(start + 1, min(len(lines), start + 6) + 1):
            pieces.append(lines[end - 1])

            if _metadata_line_key(" ".join(pieces)) == title_key:
                return (start, end)

    return None


def guess_title_from_first_page(text: str) -> str | None:
    """Guess a paper title from early first-page text.

    This heuristic starts from the first substantial non-noise line, then
    conservatively joins immediately following lines when the first line looks
    syntactically incomplete, as in titles that wrap after words like ``of``.
    """

    lines = first_nonempty_lines(text)

    for i, line in enumerate(lines):
        if looks_like_header_noise(line):
            continue

        if len(line) < 8:
            continue

        title_lines = [line]

        for candidate in lines[i + 1:]:
            if not _looks_like_title_continuation(title_lines[-1], candidate):
                break

            title_lines.append(candidate)

        return clean_title_metadata_text(" ".join(title_lines))

    return None


def guess_authors_from_first_page(
    text: str,
    title: str | None,
) -> str | None:
    """Guess an author line from the first page.

    This uses the line after the guessed title block when it looks author-like.
    Multi-line titles are treated as one title block so the final title line is
    not misclassified as the author.
    """

    if title is None:
        return None

    lines = first_nonempty_lines(text)
    title_span = _find_title_line_span(lines, title)

    if title_span is None:
        return None

    _, title_end = title_span

    if title_end >= len(lines):
        return None

    candidate = lines[title_end]

    if looks_like_header_noise(candidate):
        return None

    lower_candidate = candidate.lower()

    if "abstract" in lower_candidate:
        return None

    if _looks_like_affiliation_line(candidate):
        return None

    if len(candidate) > 200:
        return None

    return clean_author_metadata_text(candidate)


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

    title = clean_title_metadata_text(title)

    return ExtractedMetadata(
        title=title,
        year=year,
    )


def _crossref_author_name(author: dict) -> str | None:
    """Convert one Crossref author object into a display name."""

    given = clean_author_metadata_text(author.get("given"))
    family = clean_author_metadata_text(author.get("family"))

    if given is not None and family is not None:
        return f"{given} {family}"

    if family is not None:
        return family

    if given is not None:
        return given

    return clean_author_metadata_text(author.get("name"))


def _crossref_year(work: dict) -> int | None:
    """Extract a publication year from Crossref work metadata."""

    for key in [
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
        "deposited",
    ]:
        date_info = work.get(key)

        if not isinstance(date_info, dict):
            continue

        date_parts = date_info.get("date-parts")

        if not date_parts:
            continue

        first_date = date_parts[0]

        if not first_date:
            continue

        year = first_date[0]

        if year is None:
            continue

        try:
            return int(year)
        except (TypeError, ValueError):
            continue

    return None


def metadata_from_crossref_work(work: dict) -> ExtractedMetadata:
    """Normalize one Crossref work record into ExtractedMetadata."""

    titles = work.get("title") or []
    title = clean_title_metadata_text(titles[0]) if titles else None

    author_names = [
        name
        for name in (
            _crossref_author_name(author)
            for author in work.get("author", [])
        )
        if name is not None
    ]

    authors = clean_author_metadata_text(", ".join(author_names)) if author_names else None
    doi = clean_metadata_text(work.get("DOI"))

    return ExtractedMetadata(
        title=title,
        authors=authors,
        year=_crossref_year(work),
        doi=doi,
    )


def lookup_crossref_metadata(
    doi: str,
    crossref_mailto: str | None = None,
    timeout: float = 10.0,
) -> ExtractedMetadata:
    """Look up bibliographic metadata for a DOI using Crossref."""

    url = f"{CROSSREF_API_BASE_URL}/{quote(doi, safe='')}"

    if crossref_mailto is not None:
        url = f"{url}?{urlencode({'mailto': crossref_mailto})}"

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "kurrent DOI metadata lookup",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return metadata_from_crossref_work(payload.get("message", {}))
    except (
        HTTPError,
        URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return ExtractedMetadata()


def merge_metadata(
    primary: ExtractedMetadata,
    fallback: ExtractedMetadata,
) -> ExtractedMetadata:
    """Fill missing metadata fields in primary from fallback."""

    return ExtractedMetadata(
        title=clean_title_metadata_text(primary.title or fallback.title),
        authors=clean_author_metadata_text(primary.authors or fallback.authors),
        year=primary.year or fallback.year,
        doi=primary.doi or fallback.doi,
    )


def extract_metadata(
    pdf_path: str | Path,
    doi_lookup: bool = False,
    crossref_mailto: str | None = None,
) -> ExtractedMetadata:
    """Extract best-effort bibliographic metadata for a PDF."""

    pdf_path = Path(pdf_path)

    embedded = extract_embedded_metadata(pdf_path)
    early_text = extract_text_from_first_pages(pdf_path, max_pages=2)

    doi = extract_doi(early_text)
    year = extract_year(early_text)

    crossref = ExtractedMetadata()
    if doi_lookup and doi is not None:
        crossref = lookup_crossref_metadata(
            doi,
            crossref_mailto=crossref_mailto,
        )

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

    metadata = merge_metadata(crossref, embedded)
    metadata = merge_metadata(metadata, text_guess)
    metadata = merge_metadata(metadata, filename_guess)

    return ExtractedMetadata(
        title=clean_title_metadata_text(metadata.title),
        authors=clean_author_metadata_text(metadata.authors),
        year=metadata.year,
        doi=metadata.doi,
    )


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
