"""Detect and represent section structure in PDFs.

This module owns heading detection and conversion from reviewed headings to
SectionSpan objects. It deliberately contains no user-prompting code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import pymupdf

from kurrent.file_utils import silence_mupdf_messages
from kurrent.schema import SectionLine, SectionSpan


@dataclass(frozen=True, slots=True)
class HeadingCandidate:
    """A possible section heading anchored to extracted PDF text."""

    # ID assigned by kurrent before any LLM call. The LLM must choose from
    # these exact IDs, not invent its own.
    candidate_id: int

    # 0-based index into the normalized extracted-text line stream for the
    # whole PDF. This is the stable anchor used to turn an accepted candidate
    # back into a SectionSpan boundary.
    line_index: int

    # 1-based PDF page number where the candidate's source line appears.
    page: int

    # The actual normalized line extracted from the PDF. This should remain the
    # original source line even if candidate_text points to only part of it.
    line_text: str

    # Nearby normalized lines before line_text, used as context for an LLM or
    # for diagnostics.
    previous_lines: list[str]

    # Nearby normalized lines after line_text, used as context for an LLM or
    # for diagnostics.
    next_lines: list[str]

    # Lightweight deterministic diagnostics about the source line. This should
    # not be treated as authoritative; it is mainly useful for debugging.
    features: dict[str, object]

    # Optional precomputed heading-like substring to ask the LLM to judge. When
    # None, llm_sectioner.py should derive candidate_text from line_text.
    # Example:
    #   line_text: "models ... To determine whether 6 Related Work"
    #   candidate_text: "6 Related Work"
    candidate_text: str | None = None


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
    "data, materials, and software availability",
    "a model of gossip, reputations, and social behavior",
    "a model of gossip, reputations, and social",
    "data",
    "experimental design",
    "experiment design",
    "game setup",
    "strategies",
    "behavioral dimensions",
    "role of reputation",
    "influence of ordering in partner switching",
    "effects of the temptation to defect and the average degree",
    "time evolution",
    "llm setup",
    "prompting",
    "meta-prompting",
    "behavioral profiling",
    "llm's prompt comprehension",
    "llm’s prompt comprehension",
    "effect of memory window size",
    "behavioral patterns",
    "probability of cooperation",
    "sfem profile",
    "behavioral profile",
    "llms as agents",
    "llms in game theory",
    "results",
    "numerical simulation results",
    "simulation results",
    "analysis",
    "discussion",
    "discussion and conclusion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "code and data availability",
    "ethical impact",
    "paper checklist",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "prompts and their variations",
    "effect of temperature",
    "supplementary material",
    "supporting information",
}

UNNUMBERED_EXTRACTABLE_HEADINGS = {
    "abstract",
    "references",
    "bibliography",
    "works cited",
    "literature cited",
    "appendix",
    "appendices",
    "acknowledgments",
    "acknowledgements",
    "code and data availability",
    "ethical impact",
}


EMBEDDED_UNNUMBERED_EXTRACTABLE_HEADINGS = {
    "results",
    "discussion",
    "materials and methods",
    "data, materials, and software availability",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "appendices",
}


BAD_HEADING_PATTERNS = [
    r"^doi\b",
    r"^https?://",
    r"^www\.",
    r"^citation:",
    r"^editor:",
    r"^data availability statement\b",
    r"^funding\b",
    r"^competing interests\b",
    r"^significance$",
    r"^author affiliations:",
    r"^author contributions:",
    r"^\d+\s+[a-z]\s+(and|or)\s+[a-z]\)",
    r"^\d+\.\s+[a-z]\.\s+",
    r"^[a-z]\.\s+[a-z]\.\s+",
    r"^[a-z]\.\s+(theor|rev|proc|soc|evol|hum|behav|biol|sci)\b",
    r"^copyright\b",
    r"personal use is also permitted",
    r"republication/redistribution",
    r"ieee permission",
    r"^see http://www\.ieee\.org",
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
    r"^\d{3,}\s+volume\s+\d+,\s+\d{4}$",
    r"^volume\s+\d+,\s+\d{4}\s+\d{3,}$",
    r"^[a-z]\.\s+.+? et al\.:",
    r"^[a-z]\.\s+[a-z].+?:",
    r"^figure\s+\d+",
    r"^fig\.\s+\d+",
    r"^table\s+\d+",
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

    # PDF extraction can collapse the bibliography into one long line. Extract
    # only citation-entry-like spans, not every inline citation. In ordinary
    # prose, an inline citation such as "group selection [10]--[12] and" may
    # appear in the middle of a sentence; that should not count as a reference
    # entry. A bracketed reference entry should look more like:
    #
    #   [10] L. Nunney, Group selection, altruism, ...
    #
    # i.e., a bracketed number followed by author/title-looking text.
    units.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"(?:^|\s)(\[\d+\]\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’\-]+.{20,})",
            collapsed,
        )
    )

    bracketed_reference_entries = 0
    numbered_reference_entries = 0

    for unit in units:
        if re.match(
            r"^\[\d+\]\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’\-]+",
            unit,
        ):
            bracketed_reference_entries += 1
        elif re.match(
            r"^\d+\.\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’\-]+",
            unit,
        ):
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
    """Return whether a chunk-like object appears to be reference material.

    Prefer explicit section boundaries over text-density heuristics. Named
    content sections such as "Introduction" or "Discussion" often contain
    many inline citations, and two-column PDF extraction can make those
    citations look deceptively reference-like. Therefore, only use the fallback
    text heuristic for chunks that are not already assigned to a named,
    non-reference section.
    """

    section_title = getattr(chunk, "section_title", None)

    if is_reference_section_title(section_title):
        return True

    if section_title is not None and str(section_title).strip():
        return False

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



def strip_extraction_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts glued to heading text."""

    text = normalize_line(text)

    # NIH/PMC manuscript PDFs often glue side-margin labels to headings or
    # body text:
    #   II. The ModelManuscript
    #   A. Time evolutionManuscript We first study...
    #   IV. DiscussionAuthor
    #   III. Simulation Results and DiscussionsNIH-PA
    #
    # Remove both suffix occurrences and mid-line glued occurrences.
    artifact_words = [
        "NIH-PA",
        "NIH PA",
        "Author",
        "Manuscript",
    ]

    for artifact in artifact_words:
        text = re.sub(
            rf"(?<=[A-Za-z]){re.escape(artifact)}(?=\s|$)",
            " ",
            text,
        )

    changed = True

    while changed:
        changed = False

        for artifact in artifact_words:
            if text.endswith(artifact):
                text = text[: -len(artifact)].strip()
                changed = True

    return normalize_line(text)



def known_heading_prefix_from_rest(rest: str) -> str | None:
    """Return the longest known heading title at the start of rest.

    The heading may be followed by whitespace, punctuation, or glued body text.
    For example, this should return ``ACKNOWLEDGMENTS`` from:

        ACKNOWLEDGMENTS. We thank ...
    """

    rest = normalize_line(rest)
    rest_lower = rest.lower()

    for heading in sorted(COMMON_SECTION_HEADINGS, key=len, reverse=True):
        pattern = re.escape(heading).replace(r"\ ", r"\s+")
        match = re.match(
            rf"^(?P<title>{pattern})(?=$|\s|[.:;,-])",
            rest_lower,
        )

        if match is not None:
            return rest[:match.end("title")]

    return None


def extract_line_initial_lettered_heading_text(line: str) -> str | None:
    """Return a cleaned A./B./C. heading from the start of a line.

    This handles cases such as:

        A. Time evolutionManuscript We first study...
        B. Effects of the temptation to defect and the average degree

    The sectioner can then pass that cleaned candidate_text to the LLM instead
    of relying on llm_sectioner.py to infer the correct prefix.
    """

    line = strip_extraction_artifacts(normalize_line(line))

    if looks_like_bad_heading(line):
        return None

    match = re.match(r"^(?P<number>[A-Z])\.\s+(?P<rest>.+)$", line)

    if match is None:
        return None

    number = match.group("number")
    rest = match.group("rest")

    known_prefix = known_heading_prefix_from_rest(rest)

    if known_prefix is not None:
        return f"{number}. {normalize_line(known_prefix)}"

    # Fallback: take words until the first strongly body-like transition.
    # Lowercase stopwords are allowed inside headings.
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "under",
        "via",
        "with",
        "without",
    }

    words = rest.split()
    kept: list[str] = []

    for word in words:
        stripped = word.strip("()[]{}.,;:")

        if not stripped:
            continue

        lower = stripped.lower()
        titleish = stripped[0].isupper() or stripped[0].isdigit()

        if kept and not titleish and lower not in stopwords:
            break

        kept.append(word)

        if len(kept) >= 12:
            break

    if not kept:
        return None

    return f"{number}. {normalize_line(' '.join(kept))}"


def starts_with_common_section_heading(line: str) -> bool:
    """Return whether line starts with a known unnumbered section heading."""

    line = normalize_line(line)
    lowered = line.lower()

    for heading in sorted(UNNUMBERED_EXTRACTABLE_HEADINGS, key=len, reverse=True):
        pattern = re.escape(heading).replace(r"\ ", r"\s+")

        if re.match(rf"^{pattern}(?=$|\s|[.:;,-])", lowered):
            return True

    return False


def common_section_heading_prefix(line: str) -> str | None:
    """Return the known common heading at the start of line, if any.

    This can produce a clean candidate_text from glued lines like:

        Abstract the behavioral dynamics...
        ACKNOWLEDGMENTS. We thank ...

    by returning only the heading prefix.
    """

    line = normalize_line(line)

    if not line or not line[0].isupper():
        return None

    known_prefix = known_heading_prefix_from_rest(line)

    if known_prefix is not None:
        return known_prefix

    return None



def looks_like_heading(line: str) -> bool:
    """Return whether a line looks like a section heading.

    This is the older, stricter non-LLM heading detector used by
    detect_heading_candidates(). The LLM-assisted path uses the broader
    detect_heading_candidates_with_context() function instead.
    """

    line = normalize_line(line)

    if len(line) < 3 or len(line) > 120:
        return False

    if looks_like_bad_heading(line):
        return False

    if looks_like_author_or_affiliation_line(line):
        return False

    if normalize_section_title(line) in COMMON_SECTION_HEADINGS:
        return True

    if looks_like_numbered_heading(line):
        return True

    return False

def looks_like_numbered_heading(line: str) -> bool:
    """Return whether a line looks like a numbered section heading.

    This is the stricter rule used by the non-LLM heading path.
    """

    line = normalize_line(line)

    patterns = [
        r"^[IVXLCDM]+\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",
        r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-\']+$",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def looks_like_llm_numbered_candidate(line: str) -> bool:
    """Return whether line may contain a numbered or lettered heading.

    This is intentionally broader than looks_like_numbered_heading because the
    LLM path can adjudicate and clean noisy candidates. It is designed to catch
    lines like:

        3 Experimental Design is essential for a player...
        III. NUMERICAL SIMULATION RESULTS At first, we plot...
        IV. DISCUSSION In summary, we explore...
        A. Time evolutionManuscript
        B. Effects of the temptation to defect and the average degree

    The LLM receives both the raw line and context, then extracts only the real
    heading.
    """

    line = strip_extraction_artifacts(normalize_line(line))

    if len(line) < 3 or len(line) > 220:
        return False

    if looks_like_bad_heading(line):
        return False

    if looks_like_author_or_affiliation_line(line):
        return False

    patterns = [
        # Roman numeral headings, with or without a period.
        r"^[IVXLCDM]+\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-'’]+",
        # Integer and decimal section numbering.
        r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-'’]+",
        # Lettered subsection / appendix-style headings.
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-'’]+",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def looks_like_llm_heading_candidate(line: str) -> bool:
    """Return whether line should be sent as a possible LLM heading candidate.

    This favors recall over precision. Later deterministic filters in
    llm_sectioner.py and the LLM itself can reject noisy candidates, but neither
    can recover headings that never become candidates.
    """

    line = normalize_line(line)

    if len(line) < 3 or len(line) > 220:
        return False

    if looks_like_bad_heading(line):
        return False

    if looks_like_author_or_affiliation_line(line):
        return False

    if starts_with_common_section_heading(line):
        return True

    if looks_like_llm_numbered_candidate(line):
        return True

    if extract_embedded_heading_texts(line):
        return True

    return False


def normalize_common_heading_match(text: str) -> str:
    """Normalize text for common-heading substring matching."""

    text = normalize_line(text).lower()
    text = text.replace("’", "'")
    text = text.strip(" .:-")

    return text


def common_heading_title_pattern() -> str:
    """Return a regex alternation for known common section titles."""

    titles = sorted(COMMON_SECTION_HEADINGS, key=len, reverse=True)
    escaped_titles = [
        re.escape(title).replace("\\ ", r"\s+").replace("'", r"['’]")
        for title in titles
    ]

    return "|".join(escaped_titles)


def extract_embedded_heading_texts(line: str) -> list[str]:
    """Return heading-like substrings embedded inside a messy PDF line.

    Two-column PDF extraction often glues left-column prose to a right-column
    section heading, for example:

        models being introduced regularly. To determine whether 6 Related Work

    This function extracts the embedded heading substring:

        6 Related Work

    The full line remains the source anchor in HeadingCandidate.line_text.
    """

    line = normalize_line(line)

    if not line:
        return []

    heading_title = common_heading_title_pattern()
    candidates: list[str] = []

    patterns = [
        # Embedded integer/decimal headings:
        #   ... 6 Related Work
        #   ... 4.1 LLM's Prompt Comprehension
        rf"(?<![\w.])(?P<number>\d+(?:\.\d+)*)\.?\s+"
        rf"(?P<title>{heading_title})(?=$|\s|[.:;,\-])",
        # Embedded Roman numeral headings:
        #   ... III. NUMERICAL SIMULATION RESULTS
        rf"(?<![\w.])(?P<number>[IVXLCDM]+)\.?\s+"
        rf"(?P<title>{heading_title})(?=$|\s|[.:;,\-])",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            lookback = line[max(0, match.start() - 12):match.start()].lower()

            if re.search(r"(?:fig|figure|eq|equation)\.?\s*$", lookback):
                continue

            number = match.group("number").rstrip(".")
            title = normalize_line(match.group("title"))
            candidate_text = f"{number} {title}"

            if candidate_text not in candidates:
                candidates.append(candidate_text)

    # Some unnumbered section headings are often glued to the end of a
    # previous prose line in two-column PDFs. Examples observed in PNAS:
    #
    #   disagreements (d2 = 0). Results
    #   finite Discussion
    #   unbiased Materials and Methods
    #
    # Only extract these when they occur at the end of an extracted line. That
    # avoids obvious body prose such as "our results show ..." while still
    # recovering headings displaced by two-column extraction order.
    unnumbered_titles = sorted(
        EMBEDDED_UNNUMBERED_EXTRACTABLE_HEADINGS,
        key=len,
        reverse=True,
    )

    for title in unnumbered_titles:
        title_pattern = re.escape(title).replace("\\ ", r"\s+")
        patterns = [
            rf"(?<![A-Za-z])(?P<title>{title_pattern})[.:]?\s*$",
            rf"(?<![A-Za-z])(?P<title>{title_pattern})(?=[.:])",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                candidate_text = normalize_line(match.group("title"))

                if candidate_text not in candidates:
                    candidates.append(candidate_text)

    return candidates


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
        r"^(?P<number>[IVXLCDM]+)\.?[ ]+(?P<title>.+)$",
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


def extract_pdf_lines_with_pages_and_indexes(
    pdf_path: str | Path,
) -> list[tuple[int, int, str]]:
    """Extract normalized nonempty PDF text lines with global line indexes."""

    lines_with_pages = extract_pdf_lines_with_pages(pdf_path)

    return [
        (line_index, page_num, line)
        for line_index, (page_num, line) in enumerate(lines_with_pages)
    ]


def heading_candidate_features(line: str) -> dict[str, object]:
    """Return lightweight deterministic features for a candidate heading.

    The LLM prompt no longer needs these features, but we keep the function for
    diagnostics and possible future ranking/filtering.
    """

    normalized = normalize_line(line)

    return {
        "looks_numbered": looks_like_llm_numbered_candidate(normalized),
        "is_common_heading": starts_with_common_section_heading(normalized),
        "has_embedded_heading": bool(extract_embedded_heading_texts(normalized)),
        "char_len": len(normalized),
    }


def add_heading_candidate(
    candidates: list[HeadingCandidate],
    seen_keys: set[tuple[int, str | None]],
    line_index: int,
    page_num: int,
    line: str,
    previous_lines: list[str],
    next_lines: list[str],
    candidate_text: str | None = None,
) -> None:
    """Append a HeadingCandidate unless this anchor/text pair is duplicated."""

    if candidate_text is not None:
        candidate_text = strip_extraction_artifacts(candidate_text)

    key = (
        line_index,
        normalize_line(candidate_text).lower() if candidate_text else None,
    )

    if key in seen_keys:
        return

    seen_keys.add(key)

    candidates.append(
        HeadingCandidate(
            candidate_id=len(candidates),
            line_index=line_index,
            page=page_num,
            line_text=line,
            previous_lines=previous_lines,
            next_lines=next_lines,
            features=heading_candidate_features(line),
            candidate_text=candidate_text,
        )
    )


def detect_heading_candidates_with_context(
    pdf_path: str | Path,
    max_pages: int | None = None,
    context_lines: int = 2,
) -> list[HeadingCandidate]:
    """Return heading candidates with IDs and nearby extracted-text context.

    This LLM-oriented candidate generator intentionally favors recall over
    precision. It includes noisy but plausible heading windows so the LLM can
    adjudicate them. Obvious junk such as page footers is still removed here.

    If a line contains an embedded heading-like substring, this function emits
    a candidate with candidate_text set to that substring while preserving the
    full extracted line in line_text.
    """

    lines = extract_pdf_lines_with_pages_and_indexes(pdf_path)
    candidates: list[HeadingCandidate] = []
    seen_keys: set[tuple[int, str | None]] = set()

    for i, (line_index, page_num, line) in enumerate(lines):
        if max_pages is not None and page_num > max_pages:
            continue

        previous_lines = [
            candidate_line
            for _, _, candidate_line in lines[max(0, i - context_lines):i]
        ]
        next_lines = [
            candidate_line
            for _, _, candidate_line in lines[i + 1:i + 1 + context_lines]
        ]

        line_initial_lettered_heading = (
            extract_line_initial_lettered_heading_text(line)
        )

        if line_initial_lettered_heading is not None:
            add_heading_candidate(
                candidates=candidates,
                seen_keys=seen_keys,
                line_index=line_index,
                page_num=page_num,
                line=line,
                previous_lines=previous_lines,
                next_lines=next_lines,
                candidate_text=line_initial_lettered_heading,
            )
            continue

        embedded_headings = extract_embedded_heading_texts(line)

        if embedded_headings:
            for embedded_heading in embedded_headings:
                add_heading_candidate(
                    candidates=candidates,
                    seen_keys=seen_keys,
                    line_index=line_index,
                    page_num=page_num,
                    line=line,
                    previous_lines=previous_lines,
                    next_lines=next_lines,
                    candidate_text=embedded_heading,
                )
            continue

        common_prefix = common_section_heading_prefix(line)

        if common_prefix is not None:
            add_heading_candidate(
                candidates=candidates,
                seen_keys=seen_keys,
                line_index=line_index,
                page_num=page_num,
                line=line,
                previous_lines=previous_lines,
                next_lines=next_lines,
                candidate_text=common_prefix,
            )
            continue

        if looks_like_llm_heading_candidate(line):
            add_heading_candidate(
                candidates=candidates,
                seen_keys=seen_keys,
                line_index=line_index,
                page_num=page_num,
                line=line,
                previous_lines=previous_lines,
                next_lines=next_lines,
            )

    return candidates


def _decision_value(
    decision: Mapping[str, object] | object,
    key: str,
) -> object:
    """Get a value from a dict-like or object-like LLM decision."""

    if isinstance(decision, Mapping):
        return decision.get(key)

    return getattr(decision, key, None)


def make_section_spans_from_llm_decisions(
    pdf_path: str | Path,
    doc_id: str,
    candidates: Sequence[HeadingCandidate],
    decisions: Sequence[Mapping[str, object] | object],
) -> list[SectionSpan]:
    """Split PDF text into SectionSpan objects using LLM-selected candidates.

    The decisions are expected to refer to candidate_id values from candidates.
    Each selected candidate starts a new section, while the section_number and
    section_title come from the decision's cleaned values.
    """

    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    decision_by_line_index: dict[int, tuple[str | None, str | None]] = {}

    for decision in decisions:
        candidate_id = _decision_value(decision, "candidate_id")

        try:
            candidate_id = int(candidate_id)
        except (TypeError, ValueError):
            continue

        candidate = candidate_by_id.get(candidate_id)

        if candidate is None:
            continue

        raw_number = _decision_value(decision, "section_number")
        raw_title = _decision_value(decision, "section_title")

        section_number = (
            str(raw_number).strip()
            if raw_number is not None and str(raw_number).strip()
            else None
        )
        section_title = (
            str(raw_title).strip()
            if raw_title is not None and str(raw_title).strip()
            else None
        )

        if section_number is None and section_title is None:
            if candidate.candidate_text:
                section_number, section_title = parse_section_heading(
                    candidate.candidate_text,
                )
            else:
                section_number, section_title = parse_section_heading(
                    candidate.line_text,
                )

        decision_by_line_index[candidate.line_index] = (
            section_number,
            section_title,
        )

    lines = extract_pdf_lines_with_pages_and_indexes(pdf_path)
    sections: list[SectionSpan] = []

    current_section_index: int | None = None
    current_section_number: str | None = None
    current_section_title: str | None = None
    current_start_page: int | None = None
    current_end_page: int | None = None
    current_lines: list[SectionLine] = []
    next_section_index = 0

    def emit_current() -> None:
        text = " ".join(
            section_line.text
            for section_line in current_lines
        ).strip()

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
                lines=list(current_lines),
            )
        )

    for line_index, page_num, line in lines:
        if line_index in decision_by_line_index:
            emit_current()

            section_number, section_title = decision_by_line_index[line_index]

            current_section_index = next_section_index
            next_section_index += 1
            current_section_number = section_number
            current_section_title = section_title
            current_start_page = page_num
            current_end_page = page_num
            current_lines = [SectionLine(page=page_num, text=line)]
            continue

        if not current_lines:
            current_section_index = None
            current_section_number = None
            current_section_title = None
            current_start_page = page_num

        current_lines.append(SectionLine(page=page_num, text=line))
        current_end_page = page_num

    emit_current()

    return sections


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
    current_lines: list[SectionLine] = []
    next_section_index = 0

    def emit_current() -> None:
        text = " ".join(
            section_line.text
            for section_line in current_lines
        ).strip()

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
                lines=list(current_lines),
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
            current_lines = [SectionLine(page=page_num, text=line)]
            continue

        if not current_lines:
            # Front matter before the first accepted heading, or the whole
            # document if no headings were supplied.
            current_section_index = None
            current_section_number = None
            current_section_title = None
            current_start_page = page_num

        current_lines.append(SectionLine(page=page_num, text=line))
        current_end_page = page_num

    emit_current()

    return sections
