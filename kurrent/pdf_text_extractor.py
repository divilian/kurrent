"""Layout-aware text extraction for born-digital PDF files.

This module is intentionally separate from sectioning and chunking. Its job is
only to turn PDF word boxes into a stream of human reading-order text lines with
page provenance. The current implementation focuses on common one-column and
two-column scholarly article layouts, while filtering common non-body artifacts
such as rotated margin text, repeated manuscript boilerplate, and copyright
notices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re

import pymupdf

from kurrent.file_utils import normalize_path, silence_mupdf_messages
from kurrent.schema import SectionLine


@dataclass(frozen=True, slots=True)
class WordBox:
    """One word plus its rectangular location on a PDF page."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int | None = None
    line_no: int | None = None
    word_no: int | None = None
    source_line_text: str | None = None

    @property
    def width(self) -> float:
        """Return the width of the word box."""

        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Return the height of the word box."""

        return self.y1 - self.y0

    @property
    def x_center(self) -> float:
        """Return the horizontal midpoint of the word box."""

        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        """Return the vertical midpoint of the word box."""

        return (self.y0 + self.y1) / 2


@dataclass(frozen=True, slots=True)
class TextLine:
    """One reconstructed visual line of text on a PDF page."""

    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    column: str

    @property
    def width(self) -> float:
        """Return the width of the reconstructed line."""

        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Return the height of the reconstructed line."""

        return self.y1 - self.y0

    @property
    def y_center(self) -> float:
        """Return the vertical midpoint of the reconstructed line."""

        return (self.y0 + self.y1) / 2


@dataclass(frozen=True, slots=True)
class PageExtraction:
    """Layout-aware text extraction result for one PDF page."""

    page: int
    width: float
    height: float
    layout: str
    gutter_x: float | None
    lines: list[TextLine]
    filtered_lines: list[TextLine] = field(default_factory=list)


BOILERPLATE_LINE_PATTERNS = [
    r"^NIH Public Access$",
    r"^Author Manuscript$",
    r"^NIH-PA$",
    r"^NIH-PA Author Manuscript$",
    r"^Published in final edited form as:$",
    r"^DOI\s*:",
    r"^PACS\s+number\s*s?\s*:",
    r"^PACS\s+(number|numbers|no\.?|nos\.?)\b",
    r"^Phys Rev E Stat Nonlin Soft Matter Phys\. Author manuscript; available in PMC\b",
    r"^Phys Rev E Stat Nonlin Soft Matter Phys\. \d{4}\b",
    r"^Fu et al\. Page \d+$",
    r"^[*†‡].*@",
    r"^\d{4}-\d{4}/\d{4}/",
    r"^Copyright © \d{4}, Association for the Advancement of Artificial$",
    r"^Intelligence \(www\.aaai\.org\)\. All rights reserved\.$",
    r"^©\s*\d{4}\b",
    r"^\(c\)\s*\d{4}\b",
    r"^copyright\s*©?\s*\d{4}\b",
]
BOILERPLATE_LINE_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in BOILERPLATE_LINE_PATTERNS),
    flags=re.IGNORECASE,
)



def repair_pdf_control_glyphs(text: str) -> str:
    """Repair common PDF control-glyph punctuation from custom encodings.

    Some older scholarly PDFs encode ordinary punctuation using private or
    control-like glyph codes. PyMuPDF exposes those codes in raw text even
    though the visual PDF shows ordinary parentheses or citation brackets.
    Repair them before normal whitespace cleanup can discard or separate them.
    """

    return (
        text.replace("\x01", "(")
        .replace("\x02", ")")
        .replace("\x03", "[")
        .replace("\x04", "]")
    )

def normalize_extracted_text(text: str) -> str:
    """Normalize one extracted text fragment for display/chunking."""

    text = repair_pdf_control_glyphs(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def text_from_words(words: Sequence[WordBox]) -> str:
    """Join word boxes into one normalized visual line.

    If all words come from one PyMuPDF source line and raw source text is
    available, prefer that raw line because it preserves custom-encoded
    punctuation such as parentheses and citation brackets.
    """

    source_keys = {(word.block_no, word.line_no) for word in words}
    source_texts = {word.source_line_text for word in words if word.source_line_text}

    if len(source_keys) == 1 and len(source_texts) == 1:
        return normalize_extracted_text(next(iter(source_texts)))

    return normalize_extracted_text(" ".join(word.text for word in words))


def line_from_words(
    words: Sequence[WordBox],
    column: str,
) -> TextLine | None:
    """Return one TextLine from word boxes, or None for empty text."""

    text = text_from_words(words)

    if not text:
        return None

    return TextLine(
        page=words[0].page,
        text=text,
        x0=min(word.x0 for word in words),
        y0=min(word.y0 for word in words),
        x1=max(word.x1 for word in words),
        y1=max(word.y1 for word in words),
        column=column,
    )



def raw_source_line_texts(page) -> dict[tuple[int, int], str]:
    """Return repaired raw text keyed by PyMuPDF block and line numbers."""

    source_lines: dict[tuple[int, int], str] = {}
    raw_page = page.get_text("rawdict")

    for block_no, block in enumerate(raw_page.get("blocks", [])):
        for line_no, line in enumerate(block.get("lines", [])):
            text = "".join(
                "".join(char.get("c", "") for char in span.get("chars", []))
                for span in line.get("spans", [])
            )
            text = normalize_extracted_text(text)

            if text:
                source_lines[(block_no, line_no)] = text

    return source_lines

def extract_words_from_pdf_page(
    page,
    page_number: int,
) -> list[WordBox]:
    """Extract word boxes from one PyMuPDF page."""

    words: list[WordBox] = []
    source_lines = raw_source_line_texts(page)

    for raw_word in page.get_text("words", sort=False):
        x0, y0, x1, y1, text, block_no, line_no, word_no = raw_word[:8]
        text = normalize_extracted_text(str(text))

        if not text:
            continue

        words.append(
            WordBox(
                page=page_number,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                text=text,
                block_no=int(block_no),
                line_no=int(line_no),
                word_no=int(word_no),
                source_line_text=source_lines.get((int(block_no), int(line_no))),
            )
        )

    return words


def median(values: Sequence[float], fallback: float) -> float:
    """Return the median of values, or fallback for an empty sequence."""

    if not values:
        return fallback

    sorted_values = sorted(values)
    middle = len(sorted_values) // 2

    if len(sorted_values) % 2 == 1:
        return sorted_values[middle]

    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def is_likely_margin_artifact(
    word: WordBox,
    page_width: float,
    page_height: float,
) -> bool:
    """Return whether a word looks like rotated or edge-margin boilerplate.

    This intentionally uses conservative geometry. The main target is text such
    as NIH manuscript labels rotated into a far-left margin. Those words have
    very narrow x-ranges, tall boxes, and sit well outside the main body column.
    """

    if word.x_center > page_width * 0.12 and word.x_center < page_width * 0.88:
        return False

    narrow_edge_box = word.width <= page_width * 0.035
    tall_for_width = word.height >= max(16.0, word.width * 1.8)
    long_vertical_band = word.height >= page_height * 0.04

    return narrow_edge_box and (tall_for_width or long_vertical_band)


def filter_margin_artifact_words(
    words: Sequence[WordBox],
    page_width: float,
    page_height: float,
) -> list[WordBox]:
    """Remove words that look like rotated margin text."""

    return [
        word
        for word in words
        if not is_likely_margin_artifact(word, page_width, page_height)
    ]


def is_boilerplate_line(line: TextLine) -> bool:
    """Return whether a reconstructed line is non-body boilerplate."""

    text = normalize_extracted_text(line.text)

    if not text:
        return True

    if BOILERPLATE_LINE_RE.search(text):
        return True

    return False


def filter_boilerplate_lines(lines: Sequence[TextLine]) -> list[TextLine]:
    """Remove common headers, footers, manuscript labels, and notices."""

    return [line for line in lines if not is_boilerplate_line(line)]




def all_words_have_source_line_ids(words: Sequence[WordBox]) -> bool:
    """Return whether word boxes carry PyMuPDF block/line identifiers."""

    return all(
        word.block_no is not None and word.line_no is not None
        for word in words
    )


def group_words_by_source_lines(words: Sequence[WordBox]) -> list[list[WordBox]]:
    """Group words using PyMuPDF's own line identifiers when available."""

    grouped: dict[tuple[int, int], list[WordBox]] = {}

    for word in words:
        assert word.block_no is not None
        assert word.line_no is not None
        grouped.setdefault((word.block_no, word.line_no), []).append(word)

    rows = list(grouped.values())

    for row in rows:
        row.sort(key=lambda item: (item.x0, item.y0, item.word_no or 0))

    rows.sort(key=lambda row: (min(word.y0 for word in row), min(word.x0 for word in row)))
    return rows


def group_words_by_y_position(
    words: Sequence[WordBox],
    y_tolerance: float | None = None,
) -> list[list[WordBox]]:
    """Group words into visual rows by vertical position alone."""

    if not words:
        return []

    if y_tolerance is None:
        word_heights = [max(1.0, word.y1 - word.y0) for word in words]
        y_tolerance = max(2.0, median(word_heights, fallback=8.0) * 0.45)

    rows: list[list[WordBox]] = []

    for word in sorted(words, key=lambda item: (item.y_center, item.x0)):
        matching_row = None

        for row in rows:
            row_center = median([row_word.y_center for row_word in row], word.y_center)

            if abs(word.y_center - row_center) <= y_tolerance:
                matching_row = row
                break

        if matching_row is None:
            rows.append([word])
        else:
            matching_row.append(word)

    for row in rows:
        row.sort(key=lambda item: (item.x0, item.y0))

    rows.sort(key=lambda row: (min(word.y0 for word in row), min(word.x0 for word in row)))
    return rows

def group_words_into_visual_rows(
    words: Sequence[WordBox],
    y_tolerance: float | None = None,
) -> list[list[WordBox]]:
    """Group words into visual rows.

    When PyMuPDF gives reliable block/line identifiers, prefer those line
    groups. This avoids accidentally merging separate left/right column lines
    that happen to sit at nearly the same vertical position. Fake test boxes and
    fallback paths without source line IDs use coordinate-based grouping.
    """

    if not words:
        return []

    if all_words_have_source_line_ids(words):
        return group_words_by_source_lines(words)

    return group_words_by_y_position(words, y_tolerance=y_tolerance)

def find_likely_column_gutter(
    words: Sequence[WordBox],
    page_width: float,
    page_height: float,
) -> float | None:
    """Return likely x-position of a two-column gutter, if one is detected.

    The heuristic searches for repeated row-level whitespace gaps near the
    middle of the page. Looking row-by-row keeps occasional full-width headings,
    abstracts, or equations from erasing the gutter signal for the whole page.
    """

    if len(words) < 30:
        return None

    top_cutoff = page_height * 0.12
    bottom_cutoff = page_height * 0.92
    body_words = [
        word
        for word in words
        if top_cutoff <= word.y_center <= bottom_cutoff
    ]

    if len(body_words) < 24:
        body_words = list(words)

    rows = group_words_by_y_position(body_words)
    min_gutter_width = page_width * 0.04
    center_min = page_width * 0.34
    center_max = page_width * 0.66
    candidate_centers: list[float] = []

    for row in rows:
        if len(row) < 2:
            continue

        row = sorted(row, key=lambda word: word.x0)
        best_row_gap: tuple[float, float] | None = None

        for left, right in zip(row, row[1:]):
            gap_start = left.x1
            gap_end = right.x0
            gap_width = gap_end - gap_start
            gap_center = (gap_start + gap_end) / 2

            if gap_width < min_gutter_width:
                continue

            if not center_min <= gap_center <= center_max:
                continue

            if best_row_gap is None or gap_width > best_row_gap[0]:
                best_row_gap = (gap_width, gap_center)

        if best_row_gap is not None:
            candidate_centers.append(best_row_gap[1])

    if not candidate_centers:
        return None

    # A page may begin with a full-width title/abstract before switching to two
    # columns, so requiring support from 25% of all rows misses normal papers.
    # A lower threshold still rejects one-off gaps while detecting pages whose
    # lower half is two-column body text.
    min_support = max(4, int(len(rows) * 0.10))

    if len(candidate_centers) < min_support:
        return None

    return median(candidate_centers, fallback=page_width / 2)


def split_row_around_gutter(
    row: Sequence[WordBox],
    gutter_x: float | None,
) -> list[TextLine]:
    """Split one visual row into left/right lines when a gutter separates it."""

    if not row:
        return []

    if gutter_x is None:
        line = line_from_words(row, column="single")
        return [] if line is None else [line]

    left_words = [word for word in row if word.x_center < gutter_x]
    right_words = [word for word in row if word.x_center >= gutter_x]

    if not left_words or not right_words:
        # If the source extractor has already given us a one-sided line, do not
        # promote it to full-width merely because one long word crosses the
        # gutter by a few points. That was enough to re-interleave real two-
        # column pages. True full-width lines normally contain words on both
        # sides of the gutter and are handled below.
        #
        # Very short right-column lines can sit mostly to the left of a detected
        # gutter even though their left edge aligns with the right column. Treat
        # those near-gutter one-sided rows as right-column text so short final
        # lines such as "follows:" stay with their paragraph.
        if right_words:
            column = "right"
        elif min(word.x0 for word in row) >= gutter_x - 45:
            column = "right"
        else:
            column = "left"

        line = line_from_words(row, column=column)
        return [] if line is None else [line]

    left_edge = max(word.x1 for word in left_words)
    right_edge = min(word.x0 for word in right_words)
    gap = right_edge - left_edge

    # If both sides are present and there is a real whitespace gap around the
    # detected gutter, this row is two simultaneous column lines rather than one
    # full-width line. This is the key guard against row-interleaving.
    if gap > 8:
        lines: list[TextLine] = []
        left_line = line_from_words(left_words, column="left")
        right_line = line_from_words(right_words, column="right")

        if left_line is not None:
            lines.append(left_line)

        if right_line is not None:
            lines.append(right_line)

        return lines

    # A long final word can cross the gutter even though the visual line is
    # still a left-column line. Likewise for a long first word in the right
    # column. Treat very unbalanced near-gutter rows as the majority column, not
    # as full-width text.
    if len(left_words) >= 3 and len(right_words) <= 2:
        line = line_from_words(row, column="left")
        return [] if line is None else [line]

    if len(right_words) >= 3 and len(left_words) <= 2:
        line = line_from_words(row, column="right")
        return [] if line is None else [line]

    line = line_from_words(row, column="full")
    return [] if line is None else [line]



def merge_words_from_lines(lines: Sequence[TextLine]) -> TextLine:
    """Merge adjacent same-row TextLine objects into one TextLine."""

    text = normalize_extracted_text(" ".join(line.text for line in lines))

    return TextLine(
        page=lines[0].page,
        text=text,
        x0=min(line.x0 for line in lines),
        y0=min(line.y0 for line in lines),
        x1=max(line.x1 for line in lines),
        y1=max(line.y1 for line in lines),
        column=lines[0].column,
    )


def merge_adjacent_same_row_lines(lines: Sequence[TextLine]) -> list[TextLine]:
    """Merge same-column line fragments that occupy the same visual row.

    Some PDFs represent a section number and heading as separate internal lines
    even though they appear on the same visual baseline, e.g. "1" and
    "Introduction". This merges nearby same-row fragments without crossing a
    large inter-column gutter.
    """

    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda line: (line.y_center, line.x0))
    merged: list[TextLine] = []
    pending: list[TextLine] = []

    def flush_pending() -> None:
        if not pending:
            return
        merged.append(merge_words_from_lines(pending))
        pending.clear()

    for line in sorted_lines:
        if not pending:
            pending.append(line)
            continue

        previous = pending[-1]
        same_column = line.column == previous.column
        same_row = abs(line.y_center - previous.y_center) <= max(2.0, previous.height * 0.55)
        horizontal_gap = line.x0 - previous.x1
        close_gap = 0 <= horizontal_gap <= 80

        if same_column and same_row and close_gap:
            pending.append(line)
            continue

        flush_pending()
        pending.append(line)

    flush_pending()
    return sorted(merged, key=lambda line: (line.y0, line.x0))

def order_lines_for_reading(lines: Sequence[TextLine], gutter_x: float | None) -> list[TextLine]:
    """Return text lines in human reading order for one page."""

    if gutter_x is None:
        return sorted(lines, key=lambda line: (line.y0, line.x0))

    full_lines = sorted(
        [line for line in lines if line.column == "full"],
        key=lambda line: (line.y0, line.x0),
    )
    column_lines = [line for line in lines if line.column != "full"]

    ordered: list[TextLine] = []
    band_start = float("-inf")

    def emit_column_band(y_min: float, y_max: float) -> None:
        band = [
            line
            for line in column_lines
            if y_min <= line.y_center < y_max
        ]
        left = sorted(
            [line for line in band if line.column == "left"],
            key=lambda line: (line.y0, line.x0),
        )
        right = sorted(
            [line for line in band if line.column == "right"],
            key=lambda line: (line.y0, line.x0),
        )
        other = sorted(
            [line for line in band if line.column not in {"left", "right"}],
            key=lambda line: (line.y0, line.x0),
        )
        ordered.extend(left)
        ordered.extend(right)
        ordered.extend(other)

    for full_line in full_lines:
        emit_column_band(band_start, full_line.y_center)
        ordered.append(full_line)
        band_start = full_line.y_center

    emit_column_band(band_start, float("inf"))
    return ordered



def normalized_repetition_key(text: str) -> str:
    """Return a coarse key for detecting repeated header/footer lines."""

    text = normalize_extracted_text(text).lower()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[^a-z#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_margin_zone_line(line: TextLine, page_height: float) -> bool:
    """Return whether a line appears in the top/bottom page-margin zone."""

    return line.y_center <= page_height * 0.10 or line.y_center >= page_height * 0.90


def is_page_number_like_header_footer(line: TextLine, page_height: float) -> bool:
    """Return whether a margin line looks like a page number/header marker."""

    if not is_margin_zone_line(line, page_height):
        return False

    text = normalize_extracted_text(line.text)

    if re.fullmatch(r"Page\s+\d+", text, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"\d+", text):
        return True

    return False


def filter_repeated_margin_lines(
    pages: Sequence[PageExtraction],
    min_repetitions: int = 2,
) -> list[PageExtraction]:
    """Remove repeated top/bottom running headers and footers.

    This catches mechanically repeated margin text such as author running heads,
    page-number labels, journal footer strings, and manuscript availability
    notices. The filter is intentionally position-aware: body lines are not
    dropped merely because their text repeats.
    """

    key_counts: dict[str, int] = {}

    for page in pages:
        seen_on_page: set[str] = set()

        for line in page.lines:
            if not is_margin_zone_line(line, page.height):
                continue

            key = normalized_repetition_key(line.text)

            if len(key) < 3:
                continue

            seen_on_page.add(key)

        for key in seen_on_page:
            key_counts[key] = key_counts.get(key, 0) + 1

    filtered_pages: list[PageExtraction] = []

    for page in pages:
        kept_lines: list[TextLine] = []
        removed_lines: list[TextLine] = list(page.filtered_lines)

        for line in page.lines:
            key = normalized_repetition_key(line.text)
            repeated_margin_line = (
                is_margin_zone_line(line, page.height)
                and key_counts.get(key, 0) >= min_repetitions
            )
            page_number_marker = is_page_number_like_header_footer(line, page.height)

            if repeated_margin_line or page_number_marker:
                removed_lines.append(line)
            else:
                kept_lines.append(line)

        filtered_pages.append(
            PageExtraction(
                page=page.page,
                width=page.width,
                height=page.height,
                layout=page.layout,
                gutter_x=page.gutter_x,
                lines=kept_lines,
                filtered_lines=removed_lines,
            )
        )

    return filtered_pages


def extract_page_from_words(
    words: Sequence[WordBox],
    page_number: int,
    page_width: float,
    page_height: float,
) -> PageExtraction:
    """Extract one page's reading-order lines from word boxes."""

    body_words = filter_margin_artifact_words(words, page_width, page_height)
    gutter_x = find_likely_column_gutter(body_words, page_width, page_height)
    rows = group_words_into_visual_rows(body_words)
    unordered_lines: list[TextLine] = []

    for row in rows:
        unordered_lines.extend(split_row_around_gutter(row, gutter_x))

    unordered_lines = merge_adjacent_same_row_lines(unordered_lines)
    ordered_lines = order_lines_for_reading(unordered_lines, gutter_x)
    body_lines: list[TextLine] = []
    filtered_lines: list[TextLine] = []

    for line in ordered_lines:
        if is_boilerplate_line(line):
            filtered_lines.append(line)
        else:
            body_lines.append(line)

    layout = "two-column" if gutter_x is not None else "single-column"

    return PageExtraction(
        page=page_number,
        width=page_width,
        height=page_height,
        layout=layout,
        gutter_x=gutter_x,
        lines=body_lines,
        filtered_lines=filtered_lines,
    )


def extract_pdf_pages(pdf_path: str | Path) -> list[PageExtraction]:
    """Extract layout-aware page text from a PDF."""

    silence_mupdf_messages()
    pdf_path = normalize_path(pdf_path)
    pages: list[PageExtraction] = []

    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            rect = page.rect
            words = extract_words_from_pdf_page(page, page_index)
            pages.append(
                extract_page_from_words(
                    words=words,
                    page_number=page_index,
                    page_width=float(rect.width),
                    page_height=float(rect.height),
                )
            )

    return filter_repeated_margin_lines(pages)


def extract_pdf_lines(pdf_path: str | Path) -> list[SectionLine]:
    """Extract PDF text as page-aware SectionLine objects in reading order."""

    section_lines: list[SectionLine] = []

    for page in extract_pdf_pages(pdf_path):
        for line in page.lines:
            section_lines.append(SectionLine(page=line.page, text=line.text))

    return section_lines
