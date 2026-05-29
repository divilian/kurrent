"""Inspect layout-aware PDF text extraction output.

This playground does not ingest anything. It lets you compare kurrent's new
coordinate-based reading-order extraction against PyMuPDF's raw text output on
problem PDFs, especially two-column scholarly articles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

import pymupdf

from kurrent.file_utils import normalize_path, silence_mupdf_messages
from kurrent.pdf_text_extractor import extract_pdf_pages


def print_wrapped_line(text: str, indent: str = "") -> None:
    """Print one extracted line with gentle wrapping for narrow terminals."""

    width = 100
    print(
        textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=" " * len(indent),
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def parse_page_spec(page_spec: str | None, page_count: int) -> set[int] | None:
    """Parse a 1-based page list/range string such as '1,3-5'."""

    if page_spec is None:
        return None

    selected: set[int] = set()

    for raw_part in page_spec.split(","):
        part = raw_part.strip()

        if not part:
            continue

        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)

            if start > end:
                raise ValueError(f"Invalid page range: {part!r}")

            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))

    bad_pages = [page for page in selected if page < 1 or page > page_count]

    if bad_pages:
        raise ValueError(
            f"Page(s) out of range for {page_count}-page PDF: {bad_pages}"
        )

    return selected


def print_layout_pages(
    pdf_path: Path,
    page_spec: str | None,
    show_boxes: bool,
    show_filtered: bool,
) -> None:
    """Print layout-aware extracted lines for selected pages."""

    pages = extract_pdf_pages(pdf_path)
    selected_pages = parse_page_spec(page_spec, len(pages))

    for page in pages:
        if selected_pages is not None and page.page not in selected_pages:
            continue

        print()
        print("=" * 79)
        print(
            f"Page {page.page} | layout={page.layout} | "
            f"gutter_x={page.gutter_x if page.gutter_x is not None else 'n/a'}"
        )
        print("=" * 79)

        for i, line in enumerate(page.lines, start=1):
            if show_boxes:
                prefix = (
                    f"{i:03d}. [{line.column:>5} "
                    f"x={line.x0:.1f}-{line.x1:.1f} "
                    f"y={line.y0:.1f}-{line.y1:.1f}] "
                )
            else:
                prefix = f"{i:03d}. "

            print_wrapped_line(line.text, indent=prefix)

        if show_filtered and page.filtered_lines:
            print()
            print("Filtered lines:")
            for i, line in enumerate(page.filtered_lines, start=1):
                if show_boxes:
                    prefix = (
                        f"  {i:03d}. [{line.column:>5} "
                        f"x={line.x0:.1f}-{line.x1:.1f} "
                        f"y={line.y0:.1f}-{line.y1:.1f}] "
                    )
                else:
                    prefix = f"  {i:03d}. "

                print_wrapped_line(line.text, indent=prefix)


def print_raw_pages(pdf_path: Path, page_spec: str | None) -> None:
    """Print PyMuPDF's raw text output for selected pages."""

    silence_mupdf_messages()

    with pymupdf.open(pdf_path) as doc:
        selected_pages = parse_page_spec(page_spec, len(doc))

        for page_index, page in enumerate(doc, start=1):
            if selected_pages is not None and page_index not in selected_pages:
                continue

            print()
            print("=" * 79)
            print(f"Raw PyMuPDF text | Page {page_index}")
            print("=" * 79)
            print(page.get_text("text"))


def build_parser() -> argparse.ArgumentParser:
    """Build the playground argument parser."""

    parser = argparse.ArgumentParser(
        description="Inspect layout-aware text extraction for a PDF.",
    )
    parser.add_argument("pdf_path", type=Path, help="PDF to inspect.")
    parser.add_argument(
        "--pages",
        help="1-based pages to show, e.g. '1', '2-4', or '1,3-5'.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also show PyMuPDF's raw text extraction for comparison.",
    )
    parser.add_argument(
        "--boxes",
        action="store_true",
        help="Show reconstructed line coordinates and column labels.",
    )
    parser.add_argument(
        "--show-filtered",
        action="store_true",
        help="Show lines removed as boilerplate, margin artifacts, or headers/footers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the playground."""

    parser = build_parser()
    args = parser.parse_args(argv)
    pdf_path = normalize_path(args.pdf_path)

    if not pdf_path.is_file():
        parser.error(f"No such PDF file: {pdf_path}")

    print(f"PDF: {pdf_path}")

    try:
        print_layout_pages(
            pdf_path,
            page_spec=args.pages,
            show_boxes=args.boxes,
            show_filtered=args.show_filtered,
        )

        if args.raw:
            print_raw_pages(pdf_path, page_spec=args.pages)
    except ValueError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
