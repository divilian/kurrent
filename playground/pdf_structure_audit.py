"""Audit PDFs for section-structure usefulness.

This script helps estimate how many PDFs in a directory have:

1. a meaningful PDF outline / structural table of contents;
2. no outline, but plausible text headings we could use for section-aware chunking;
3. neither.

Run from the project root with:

    python -m playground.pdf_structure_audit /path/to/pdf/directory

Optional:

    python -m playground.pdf_structure_audit /path/to/pdf/directory --max-pages
    python -m playground.pdf_structure_audit /path/to/pdf/directory --show-all
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import pymupdf


QUIT_WITH_ERROR = 1

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
    r"^[A-Z][A-Z ,.'-]+,\s+[A-Z][A-Z ,.'-]+,\s+AND\s+[A-Z]",
    r"^\d+\s+MCS\b",
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
    r"^\d+\s*(program|center|centre|department|school|college|faculty|institute|laboratory|lab)\b",
    r"^\(?ministry of education\)?",
    r"^canada\s+[a-z]\d[a-z]\s*\d[a-z]\d$",
    r"^usa$",
]


@dataclass(slots=True)
class PdfStructureResult:
    path: Path
    bucket: str
    outline_entries: list[str]
    heading_candidates: list[str]
    pages_examined: int
    error: str | None = None


def silence_mupdf_messages() -> None:
    """Suppress noisy MuPDF parser diagnostics during this audit."""

    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)


def discover_pdfs(root: Path) -> list[Path]:
    """Return all PDFs recursively under root, or root itself if it is a PDF."""

    root = root.expanduser().resolve()

    if root.is_file():
        if root.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {root}")

        return [root]

    if not root.is_dir():
        raise FileNotFoundError(f"No such file or directory: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def normalize_line(line: str) -> str:
    """Normalize a candidate heading line."""

    return re.sub(r"\s+", " ", line).strip()


def looks_like_author_or_affiliation_line(line: str) -> bool:
    """Return whether a line looks like authors/affiliations, not a heading."""

    line = normalize_line(line)

    # Common author-list pattern:
    # Feng Fu1,2, Christoph Hauert1,3, Martin A. Nowak1,4,*, and Long Wang2,†
    if re.search(r"\b[A-Z][A-Za-z.-]+[0-9,*†‡]", line):
        if "," in line or " and " in line:
            return True

    # Common affiliation pattern:
    # 1Program for Evolutionary Dynamics, Harvard University...
    if re.match(
        r"^\d+\s*(program|center|centre|department|school|college|faculty|"
        r"institute|laboratory|lab)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    # Address-heavy lines are almost never section headings.
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

    if len(words & address_words) >= 2:
        return True

    return False


def looks_like_bad_heading(line: str) -> bool:
    """Return True for obvious header/footer/citation junk."""

    lowered = line.lower().strip()

    if not lowered:
        return True

    for pattern in BAD_HEADING_PATTERNS:
        if re.search(pattern, lowered):
            return True

    return False


def has_reasonable_title_case(line: str) -> bool:
    """Return whether the line has a title-ish capitalization pattern."""

    words = re.findall(r"[A-Za-z][A-Za-z-]*", line)

    if not words:
        return False

    # Very permissive: enough words start uppercase, or the whole line is
    # mostly uppercase. This catches "Materials and Methods" and "RESULTS".
    uppercase_initials = sum(word[0].isupper() for word in words)
    uppercase_ratio = uppercase_initials / len(words)

    letters = re.findall(r"[A-Za-z]", line)
    all_capsish = bool(letters) and (
        sum(letter.isupper() for letter in letters) / len(letters) > 0.65
    )

    return uppercase_ratio >= 0.50 or all_capsish

def looks_like_numbered_heading(line: str) -> bool:
    """Return whether the line looks like a numbered section heading."""

    line = normalize_line(line)

    patterns = [
        # I. INTRODUCTION
        # II. MODEL AND DYNAMICS
        # IV. RESULTS
        r"^[IVXLCDM]+\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",

        # 1. Introduction
        # 2.3 Simulation Results
        # 4.1.2 Robustness Checks
        r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",

        # A. Appendix Details
        # B. Additional Simulations
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",
    ]

    return any(re.match(pattern, line) for pattern in patterns)

def looks_like_heading(line: str) -> bool:
    """Return whether a line looks like a plausible section heading.

    This deliberately favors precision over recall. For this audit, a missed
    heading is less damaging than classifying author names, affiliations, and
    running headers as headings.
    """

    line = normalize_line(line)

    if looks_like_bad_heading(line):
        return False

    if looks_like_author_or_affiliation_line(line):
        return False

    if len(line) < 3 or len(line) > 120:
        return False

    lowered = line.lower()

    if lowered in COMMON_SECTION_HEADINGS:
        return True

    if looks_like_numbered_heading(line):
        return True

    return False

def get_meaningful_outline_entries(doc: pymupdf.Document) -> list[str]:
    """Return nontrivial PDF outline entries, if any."""

    toc = doc.get_toc(simple=True)

    entries = []

    for level, title, page in toc:
        title = normalize_line(title)

        if not title:
            continue

        if looks_like_bad_heading(title):
            continue

        entries.append(title)

    # A one-entry outline is often just the document title, not a useful
    # section structure.
    if len(entries) < 2:
        return []

    return entries


def extract_lines_from_first_pages(
    doc: pymupdf.Document,
    max_pages: int,
) -> tuple[list[str], int]:
    """Extract normalized text lines from the first max_pages pages."""

    lines = []
    pages_examined = min(len(doc), max_pages)

    for page_index in range(pages_examined):
        page = doc.load_page(page_index)
        text = page.get_text("text")

        for raw_line in text.splitlines():
            line = normalize_line(raw_line)

            if line:
                lines.append(line)

    return lines, pages_examined


def dedupe_preserving_order(values: list[str]) -> list[str]:
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


def find_heading_candidates(
    doc: pymupdf.Document,
    max_pages: int,
) -> tuple[list[str], int]:
    """Find plausible section headings in early extracted text."""

    lines, pages_examined = extract_lines_from_first_pages(doc, max_pages)

    candidates = [
        line
        for line in lines
        if looks_like_heading(line)
    ]

    candidates = dedupe_preserving_order(candidates)

    # Heuristic: one lonely candidate is usually not enough evidence that this
    # PDF has usable section structure.
    if len(candidates) < 2:
        return [], pages_examined

    return candidates, pages_examined


def audit_pdf(path: Path, max_pages: int) -> PdfStructureResult:
    """Classify one PDF into structural_tree, heading_candidates, or neither."""

    try:
        with pymupdf.open(path) as doc:
            outline_entries = get_meaningful_outline_entries(doc)

            if outline_entries:
                return PdfStructureResult(
                    path=path,
                    bucket="structural_tree",
                    outline_entries=outline_entries,
                    heading_candidates=[],
                    pages_examined=0,
                )

            heading_candidates, pages_examined = find_heading_candidates(
                doc,
                max_pages=max_pages,
            )

            if heading_candidates:
                return PdfStructureResult(
                    path=path,
                    bucket="heading_candidates",
                    outline_entries=[],
                    heading_candidates=heading_candidates,
                    pages_examined=pages_examined,
                )

            return PdfStructureResult(
                path=path,
                bucket="neither",
                outline_entries=[],
                heading_candidates=[],
                pages_examined=pages_examined,
            )

    except Exception as exc:
        return PdfStructureResult(
            path=path,
            bucket="error",
            outline_entries=[],
            heading_candidates=[],
            pages_examined=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def print_result_detail(result: PdfStructureResult) -> None:
    """Print one PDF's classification details."""

    print()
    print(result.path)
    print(f"  bucket: {result.bucket}")

    if result.error is not None:
        print(f"  error:  {result.error}")
        return

    if result.outline_entries:
        print("  outline entries:")
        for title in result.outline_entries[:12]:
            print(f"    - {title}")

        if len(result.outline_entries) > 12:
            print(f"    ... {len(result.outline_entries) - 12} more")

    if result.heading_candidates:
        print(f"  pages examined: {result.pages_examined}")
        print("  heading candidates:")
        for title in result.heading_candidates[:12]:
            print(f"    - {title}")

        if len(result.heading_candidates) > 12:
            print(f"    ... {len(result.heading_candidates) - 12} more")


def print_summary(results: list[PdfStructureResult]) -> None:
    """Print aggregate counts and percentages."""

    total = len(results)
    counts = Counter(result.bucket for result in results)

    print()
    print("PDF structure audit")
    print("-------------------")
    print(f"PDFs examined: {total}")
    print()

    for bucket in [
        "structural_tree",
        "heading_candidates",
        "neither",
        "error",
    ]:
        count = counts[bucket]
        percent = 100 * count / total if total else 0

        print(f"{bucket:20s} {count:5d}  {percent:6.1f}%")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PDFs for outline trees and heading candidates.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="PDF file or directory to search recursively.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Number of early pages to scan for heading candidates.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show per-PDF details for all buckets.",
    )
    parser.add_argument(
        "--show-problematic",
        action="store_true",
        help="Show per-PDF details for neither/error buckets.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_pages < 1:
        print("--max-pages must be at least 1.", file=sys.stderr)
        return QUIT_WITH_ERROR

    silence_mupdf_messages()

    pdf_paths = discover_pdfs(args.root)

    if not pdf_paths:
        print(f"No PDFs found under {args.root}")
        return 0

    results = []

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{i}/{len(pdf_paths)}] {pdf_path.name}")
        results.append(
            audit_pdf(
                pdf_path,
                max_pages=args.max_pages,
            )
        )

    print_summary(results)

    if args.show_all:
        for result in results:
            print_result_detail(result)
    elif args.show_problematic:
        for result in results:
            if result.bucket in {"neither", "error"}:
                print_result_detail(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
