"""LLM-assisted section recognition.

This module asks a local Ollama model to choose real section headings from
kurrent-generated heading candidates. The LLM does not locate headings from
scratch; it only selects from anchored candidate_id values.

Important design choice:
    A candidate_id anchors a possible heading location, but the candidate line
    may be messy PDF-extracted text. For example, a two-column PDF might yield:

        II MODEL increased only provided that one player adopts the coop-

    The prompt therefore gives the LLM both:
        - raw_line: the original extracted line
        - candidate_text: a deterministic best-effort heading prefix
        - local context before/after the candidate

    The LLM should select candidate IDs, but it may clean the final
    section_number and section_title.

Note:
    Broadening which raw lines become HeadingCandidate objects belongs in
    sectioner.py. This file can clean, filter, and present candidates better,
    but it cannot recover headings that sectioner.py never generated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any
from urllib import error, request

from kurrent.config import DEFAULT_OLLAMA_URL, DEFAULT_SECTION_RECOGNITION_LLM
from kurrent.sectioner import HeadingCandidate, parse_section_heading

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_SINGLETON_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_NUM_PREDICT",
    "DEFAULT_OLLAMA_BATCH_SIZE",
    "OllamaTimeoutError",
    "LLMSectioningUnavailableError",
    "SectionHeadingDecision",
    "candidate_to_prompt_dict",
    "filtered_candidates",
    "select_section_headings_with_ollama",
]

DEFAULT_OLLAMA_MODEL = DEFAULT_SECTION_RECOGNITION_LLM
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_OLLAMA_SINGLETON_TIMEOUT_SECONDS = 30
DEFAULT_OLLAMA_NUM_PREDICT = 512
DEFAULT_OLLAMA_BATCH_SIZE = 1
DEFAULT_OLLAMA_MAX_CONSECUTIVE_FAILURES = 2


COMMON_HEADING_TITLES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "model",
    "models",
    "method",
    "methods",
    "materials and methods",
    "data, materials, and software availability",
    "a model of gossip, reputations, and social behavior",
    "a model of gossip, reputations, and social",
    "data",
    "time evolution",
    "effects of the temptation to defect and the average degree",
    "influence of ordering in partner switching",
    "role of reputation",
    "behavioral patterns",
    "llm's prompt comprehension",
    "llm’s prompt comprehension",
    "effect of memory window size",
    "llms in game theory",
    "llms as agents",
    "behavioral profiling",
    "meta-prompting",
    "prompting",
    "llm setup",
    "behavioral dimensions",
    "strategies",
    "game setup",
    "experimental design",
    "experiment design",
    "results",
    "numerical simulation results",
    "simulation results",
    "analysis",
    "discussion",
    "discussion and conclusion",
    "conclusion",
    "conclusions",
    "related work",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "acknowledgments",
    "acknowledgements",
    "code and data availability",
    "ethical impact",
}


TITLE_STOPWORDS = {
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
    "vs",
    "with",
    "without",
}


FIGURE_CONTEXT_WORDS = {
    "accuracy",
    "comprehension",
    "confidence",
    "figure",
    "fig.",
    "model",
    "models",
    "pcoop",
    "response",
    "round",
    "rules",
    "sfem",
    "state",
    "table",
    "time",
    "window",
}


MODEL_OR_LEGEND_WORDS = {
    "gpt",
    "gpt3",
    "gpt3.5",
    "gpt3.5t",
    "llama",
    "llama2",
    "llama3",
}


class OllamaTimeoutError(RuntimeError):
    """Raised when Ollama does not answer before the request timeout."""


class LLMSectioningUnavailableError(RuntimeError):
    """Raised when Ollama failures make LLM sectioning impractical."""


class _HeadingCandidateFailedError(RuntimeError):
    """Raised internally when one candidate cannot be classified by Ollama."""


@dataclass(frozen=True, slots=True)
class SectionHeadingDecision:
    """An LLM's validated decision that a candidate is a real heading."""

    candidate_id: int
    section_number: str | None
    section_title: str
    confidence: str | None = None


def _normalize_spaces(text: str) -> str:
    """Normalize whitespace in LLM-facing strings."""

    return re.sub(r"\s+", " ", text).strip()



def _strip_extraction_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts glued to heading text."""

    text = _normalize_spaces(text)

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

    return _normalize_spaces(text)


def _normalize_for_match(text: str) -> str:
    """Normalize a string for simple lower-case matching."""

    text = _normalize_spaces(text).lower()
    text = text.strip(".:;,-")

    return text


def _looks_like_page_footer_or_running_header(line_text: str) -> bool:
    """Return whether a line is obvious page/footer/header debris."""

    line = _normalize_spaces(line_text)
    lower = line.lower()

    footer_patterns = [
        # IEEE Access style page footer:
        #   82534 VOLUME 7, 2019
        r"^\d{3,}\s+volume\s+\d+,\s+\d{4}$",
        r"^volume\s+\d+,\s+\d{4}\s+\d{3,}$",
        # Generic bare page numbers.
        r"^\d{1,5}$",
        # Running author/title lines in the Dong example.
        r"^[a-z]\.\s+.+?:\s+.+$",
        # Common conference/proceedings footer with a page number.
        r"^proceedings of .+\s+\d{1,5}$",
    ]

    return any(
        re.match(pattern, lower)
        for pattern in footer_patterns
    )


def _context_text(candidate: HeadingCandidate) -> str:
    """Return candidate context as one normalized lower-case string."""

    pieces = (
        list(candidate.previous_lines)
        + [candidate.line_text]
        + list(candidate.next_lines)
    )

    return _normalize_spaces(" ".join(pieces)).lower()


def _looks_like_chart_or_figure_debris(candidate: HeadingCandidate) -> bool:
    """Return whether a candidate is probably an axis/legend/chart label."""

    line = _normalize_spaces(candidate.line_text)
    lower = line.lower()
    context = _context_text(candidate)

    # Examples observed in Fontana 2025:
    #   1.0 Llama2
    #   0.2 Llama3
    # These look like decimal axis labels plus legend labels, not section
    # headings. True subsection numbers almost never start with 0.x or 1.0
    # followed by a model name.
    if re.match(r"^(?:0(?:\.\d+)?|1\.0)\s+[A-Za-z0-9.]+$", line):
        if any(word in lower for word in MODEL_OR_LEGEND_WORDS):
            return True

        figure_signal_count = sum(
            1
            for word in FIGURE_CONTEXT_WORDS
            if word in context
        )

        if figure_signal_count >= 2:
            return True

    # A short decimal number plus a very short label is suspicious if the
    # surrounding context is dominated by figure/table words.
    if re.match(r"^\d+\.\d+\s+\S{2,12}$", line):
        figure_signal_count = sum(
            1
            for word in FIGURE_CONTEXT_WORDS
            if word in context
        )

        if figure_signal_count >= 3:
            return True

    return False



BIBLIOGRAPHIC_SIGNALS = {
    "[pubmed",
    "pubmed",
    "doi",
    "science",
    "nature",
    "phys rev",
    "physical review",
    "proc natl acad sci",
    "pnas",
    "j theor biol",
    "theor biol",
    "ecol lett",
    "biol lett",
    "new j phys",
    "europhys lett",
    "epl",
    "complexity",
    "am j phys",
    "j. theor. biol",
    "proc. natl. acad. sci",
    "cambridge university press",
    "harvard university press",
    "princeton university press",
    "basic books",
    "university press",
    "vol.",
    "ibid",
}


def _looks_like_numbered_reference_entry(text: str) -> bool:
    """Return whether text looks like a numbered bibliography entry.

    This deliberately requires a leading reference number plus bibliographic
    evidence. We do not want to reject ordinary numbered section headings such
    as "2. Model" or "3. Experimental Design" merely because they begin with
    a number.
    """

    text = _normalize_spaces(text)

    if not re.match(r"^\d+\.?\s+", text):
        return False

    lower = text.lower()

    if any(signal in lower for signal in BIBLIOGRAPHIC_SIGNALS):
        return True

    # Compact journal reference pattern, e.g.:
    #   5. Doebeli M, Hauert C. Ecol Lett 2005;8:748.
    #   18. Szabó G, Hauert C. Phys Rev Lett 2002;89:118101.
    if re.search(r"\b(?:19|20)\d{2};\s*\d+", text):
        return True

    # Book-ish reference entries often start with an author, title, publisher,
    # and year; this catches examples such as:
    #   3. Smith, J Maynard. Evolution and the theory of games. Cambridge...
    if re.search(r"\b(?:19|20)\d{2}\b", text) and re.search(
        r"\b(?:press|books|publisher|university)\b",
        lower,
    ):
        return True

    # Author-list-ish start plus a year somewhere later. This avoids rejecting
    # real headings like "2. Background" because they do not contain a year.
    if re.match(
        r"^\d+\.?\s+"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
        r"(?:\s+[A-Z]{1,4}\.)?"
        r"(?:,|\.)\s+",
        text,
    ):
        return bool(re.search(r"\b(?:19|20)\d{2}\b", text))

    return False


def _looks_like_unnumbered_reference_entry_fragment(text: str) -> bool:
    """Return whether text looks like a bibliography fragment candidate.

    Two-column/reference extraction can produce candidate_text without the
    leading reference number, e.g. "Annu Rev Ecol Syst" or "Vukov J...".
    This helper is intentionally conservative and should mainly be used in
    combination with the raw line, which often still contains the leading
    reference number.
    """

    text = _normalize_spaces(text)
    lower = text.lower()

    if any(signal in lower for signal in BIBLIOGRAPHIC_SIGNALS):
        return True

    if re.search(r"\b(?:19|20)\d{2};\s*\d+", text):
        return True

    # Author-list-ish fragment such as "Vukov J, Szabó G. Phys Rev E...".
    if re.match(
        r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+\s+[A-Z]{1,4},\s+"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+",
        text,
    ) and re.search(r"\b(?:19|20)\d{2}\b", text):
        return True

    return False


def _looks_like_reference_entry_candidate(candidate: HeadingCandidate) -> bool:
    """Return whether a candidate appears to be a reference-list entry."""

    candidate_text = _candidate_text_for_filtering(candidate)
    raw_line = _normalize_spaces(candidate.line_text)

    if _looks_like_numbered_reference_entry(candidate_text):
        return True

    if _looks_like_numbered_reference_entry(raw_line):
        return True

    # If the raw line is clearly a numbered reference entry, reject even if the
    # cleaned candidate_text lost the leading number and now looks like a title.
    if re.match(r"^\d+\.?\s+", raw_line):
        return _looks_like_unnumbered_reference_entry_fragment(candidate_text)

    return False

def _should_send_candidate_to_llm(candidate: HeadingCandidate) -> bool:
    """Return whether a candidate should be included in the LLM payload."""

    if _looks_like_page_footer_or_running_header(candidate.line_text):
        return False

    if _looks_like_chart_or_figure_debris(candidate):
        return False

    if _looks_like_reference_entry_candidate(candidate):
        return False

    return True


def _title_case_prefix(rest: str, max_words: int = 10) -> str:
    """Return a plausible title-case heading prefix from remaining text.

    This keeps title stopwords such as "of" and "and", so:

        Effect of Memory Window Size

    does not get truncated to:

        Effect
    """

    words = rest.split()
    kept: list[str] = []

    for word in words:
        stripped = word.strip("()[]{}.,;:")

        if not stripped:
            continue

        lower = stripped.lower()
        starts_titleish = stripped[0].isupper() or stripped[0].isdigit()

        if kept and not starts_titleish and lower not in TITLE_STOPWORDS:
            break

        kept.append(word)

        if len(kept) >= max_words:
            break

    return _strip_extraction_artifacts(_normalize_spaces(" ".join(kept)))


def _clean_heading_prefix(line_text: str) -> str:
    """Return a deterministic best-effort heading prefix.

    This tries to rescue common headings from PDF lines where the heading has
    been glued to body text, such as:

        II MODEL increased only provided that one player adopts the coop-

    which should become:

        II MODEL

    The raw extracted line is still preserved separately for the LLM.
    """

    line = _strip_extraction_artifacts(_normalize_spaces(line_text))

    if not line:
        return line

    # Roman numeral section headings, with or without a period:
    #
    #   I. INTRODUCTION
    #   II MODEL increased only provided ...
    #   III NUMERICAL SIMULATION RESULTS At first, we plot ...
    roman_match = re.match(
        r"^(?P<number>[IVXLCDM]+)\.?\s+(?P<rest>.+)$",
        line,
    )

    if roman_match is not None:
        number = roman_match.group("number")
        rest = roman_match.group("rest")
        rest_lower = rest.lower()

        for title in sorted(COMMON_HEADING_TITLES, key=len, reverse=True):
            if rest_lower == title or rest_lower.startswith(title):
                title_text = rest[:len(title)]
                return f"{number} {_normalize_spaces(title_text)}"

        # Otherwise keep an initial all-caps run. This catches headings like:
        #   IV DISCUSSION In summary, ...
        caps_match = re.match(
            r"^(?P<title>[A-Z][A-Z0-9&/,\- ]{2,})(?:\s+[a-z].*)?$",
            rest,
        )

        if caps_match is not None:
            title_text = _normalize_spaces(caps_match.group("title"))
            return f"{number} {title_text}"

    # Decimal or integer-numbered headings:
    #
    #   3 Experimental Design
    #   3.1 LLM Setup erated games ...
    #   4.2 Effect of Memory Window Size
    numbered_match = re.match(
        r"^(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<rest>.+)$",
        line,
    )

    if numbered_match is not None:
        number = numbered_match.group("number").rstrip(".")
        rest = numbered_match.group("rest")
        rest_lower = rest.lower()

        for title in sorted(COMMON_HEADING_TITLES, key=len, reverse=True):
            if rest_lower == title or rest_lower.startswith(title):
                title_text = rest[:len(title)]
                return f"{number} {_normalize_spaces(title_text)}"

        title_prefix = _title_case_prefix(rest)

        if title_prefix:
            return f"{number} {title_prefix}"

    # Lettered subsection / appendix-style headings:
    #
    #   A. Time evolutionManuscript We first study...
    #   B. Effects of the temptation to defect and the average degree
    lettered_match = re.match(
        r"^(?P<number>[A-Z])\.\s+(?P<rest>.+)$",
        line,
    )

    if lettered_match is not None:
        number = lettered_match.group("number")
        rest = lettered_match.group("rest")
        rest_lower = rest.lower()

        for title in sorted(COMMON_HEADING_TITLES, key=len, reverse=True):
            if rest_lower == title or rest_lower.startswith(title + " "):
                title_text = rest[:len(title)]
                return f"{number}. {_normalize_spaces(title_text)}"

        title_prefix = _title_case_prefix(rest)

        if title_prefix:
            return f"{number}. {title_prefix}"

    # Unnumbered common headings embedded in an otherwise clean line.
    normalized = _normalize_for_match(line)

    if normalized in COMMON_HEADING_TITLES:
        return line

    return line


def candidate_to_prompt_dict(candidate: HeadingCandidate) -> dict[str, Any]:
    """Convert a HeadingCandidate into a compact JSON-serializable dict."""

    raw_line = _normalize_spaces(candidate.line_text)

    if getattr(candidate, "candidate_text", None) is None:
        candidate_text = _clean_heading_prefix(raw_line)
    else:
        candidate_text = _strip_extraction_artifacts(
            _normalize_spaces(candidate.candidate_text)
        )

    return {
        "candidate_id": candidate.candidate_id,
        "page": candidate.page,
        "candidate_text": candidate_text,
        "raw_line": raw_line,
        "previous_lines": [
            _normalize_spaces(line)
            for line in candidate.previous_lines
        ],
        "next_lines": [
            _normalize_spaces(line)
            for line in candidate.next_lines
        ],
    }


def _candidate_text_for_filtering(candidate: HeadingCandidate) -> str:
    """Return the exact candidate_text that would be sent to the LLM."""

    raw_line = _normalize_spaces(candidate.line_text)

    if getattr(candidate, "candidate_text", None) is None:
        return _clean_heading_prefix(raw_line)

    return _strip_extraction_artifacts(
        _normalize_spaces(candidate.candidate_text)
    )


def _candidate_text_is_heading_like(candidate_text: str) -> bool:
    """Return whether candidate_text is plausible heading text."""

    text = _normalize_spaces(candidate_text)
    lowered = text.lower().strip(" .:")

    if not text:
        return False

    if lowered in COMMON_HEADING_TITLES:
        # Avoid treating body-line starts like "models ..." or "analysis ..."
        # as standalone headings after cleanup.
        if lowered in {
            "model",
            "models",
            "analysis",
            "strategies",
            "behavioral dimensions",
            "prompting",
        }:
            return False
        return True

    # Numbered headings; reject impossible-looking huge section numbers.
    number_match = re.match(
        r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+)$",
        text,
    )

    if number_match is not None:
        number_text = number_match.group("number")
        first_number = int(number_text.split(".")[0])

        if first_number == 0 or first_number > 50:
            return False

        if number_text.endswith(".0"):
            return False

        title = _normalize_spaces(number_match.group("title"))

        if title and title[0].isupper():
            return True

    roman_match = re.match(
        r"^[IVXLCDM]+\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-'’]+$",
        text,
    )

    if roman_match is not None:
        return True

    lettered_match = re.match(
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-'’]+$",
        text,
    )

    if lettered_match is not None:
        return True

    return False



def _section_level(candidate_text: str) -> int:
    """Return a rough nesting level for a section candidate."""

    text = _normalize_spaces(candidate_text)

    if re.match(r"^[A-Z]\.\s+", text):
        return 3

    decimal_match = re.match(r"^\d+(?:\.\d+)+\s+", text)

    if decimal_match is not None:
        return 2

    top_level_match = re.match(r"^(?:\d+|[IVXLCDM]+)\.?\s+", text)

    if top_level_match is not None:
        return 1

    return 0


def _group_candidates_for_prompt(
    candidates: list[HeadingCandidate],
) -> list[dict[str, object]]:
    """Return compact grouped candidate dicts for the LLM prompt."""

    grouped: dict[tuple[int, str], dict[str, object]] = {}

    for candidate in filtered_candidates(candidates):
        prompt_dict = candidate_to_prompt_dict(candidate)
        candidate_text = _normalize_spaces(str(prompt_dict["candidate_text"]))

        key = (candidate.page, candidate_text.lower())

        item = grouped.get(key)

        if item is None:
            grouped[key] = {
                "candidate_id": candidate.candidate_id,
                "also_candidate_ids": [],
                "page": candidate.page,
                "candidate_text": candidate_text,
                "raw_line": prompt_dict["raw_line"],
                "previous_line": (
                    prompt_dict["previous_lines"][-1]
                    if prompt_dict["previous_lines"]
                    else ""
                ),
                "next_line": (
                    prompt_dict["next_lines"][0]
                    if prompt_dict["next_lines"]
                    else ""
                ),
                "level_hint": _section_level(candidate_text),
            }
            continue

        item["also_candidate_ids"].append(candidate.candidate_id)

    return list(grouped.values())


def filtered_candidates(
    candidates: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    """Return candidates after deterministic noise filtering and dedupe."""

    filtered: list[HeadingCandidate] = []
    seen_texts: set[str] = set()

    for candidate in candidates:
        if not _should_send_candidate_to_llm(candidate):
            continue

        candidate_text = _candidate_text_for_filtering(candidate)

        if not _candidate_text_is_heading_like(candidate_text):
            continue

        key = candidate_text.lower()

        if key in seen_texts:
            continue

        seen_texts.add(key)
        filtered.append(candidate)

    return filtered


def _build_section_heading_prompt(
    candidates: list[HeadingCandidate],
) -> str:
    """Build the user prompt for LLM heading adjudication."""

    candidate_payload = _group_candidates_for_prompt(candidates)
    example_payload = {
        "section_headings": [
            {
                "candidate_id": 0,
                "section_number": "1",
                "section_title": "Introduction",
                "confidence": "high",
            }
        ]
    }

    return (
        "You are helping kurrent identify real section headings in an "
        "academic PDF. You will receive candidate heading locations extracted "
        "from the PDF. Some candidates are real headings; some are junk, "
        "author/affiliation lines, headers/footers, figure text, chart labels, "
        "or headings with extra body text accidentally glued on.\n\n"
        "Each candidate has a candidate_id. Use candidate_id values exactly "
        "as provided. Do not invent candidate IDs.\n\n"
        "candidate_text is a deterministic best-effort cleaned heading prefix. "
        "raw_line is the original PDF-extracted line. If raw_line contains "
        "body text glued after the heading, extract only the heading itself. "
        "For example, if candidate_text is 'II MODEL' and raw_line is "
        "'II MODEL increased only provided that...', the correct "
        "section_number is 'II' and the correct section_title is 'MODEL'.\n\n"
        "Reject page numbers, running headers/footers, chart axis labels, "
        "legend labels, figure/table labels, author lines, and affiliations. "
        "Do not classify chart labels like '1.0 Llama2' or '0.2 Llama3' as "
        "section headings.\n\n"
        "Select only candidates that are actual section headings in the "
        "paper. Clean the section_number and section_title. Do not include "
        "extra body text in section_title.\n\n"
        "Return ONLY valid JSON with this exact top-level shape:\n"
        f"{json.dumps(example_payload, ensure_ascii=False, indent=2)}\n\n"
        "If no candidates are real headings, return "
        "{\"section_headings\": []}.\n\n"
        "Candidates:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
    )


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence if the LLM added one."""

    text = text.strip()

    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match is None:
        return text

    return match.group(1).strip()


def _extract_json_payload(response_text: str) -> str:
    """Extract a JSON object/list from an LLM response."""

    response_text = _strip_code_fence(response_text).strip()

    if not response_text:
        raise ValueError("Ollama returned an empty response.")

    if response_text[0] in "{[":
        return response_text

    object_start = response_text.find("{")
    list_start = response_text.find("[")

    starts = [
        start
        for start in [object_start, list_start]
        if start != -1
    ]

    if not starts:
        preview = response_text[:500]
        raise ValueError(
            "LLM response did not contain JSON. Raw response preview:\n"
            f"{preview}"
        )

    start = min(starts)
    opener = response_text[start]
    closer = "}" if opener == "{" else "]"
    end = response_text.rfind(closer)

    if end == -1 or end <= start:
        preview = response_text[:500]
        raise ValueError(
            "LLM response looked like it contained JSON, but no complete "
            "JSON payload could be found. Raw response preview:\n"
            f"{preview}"
        )

    return response_text[start:end + 1]


def _parse_section_heading_response(
    response_text: str,
    valid_candidate_ids: set[int],
) -> list[SectionHeadingDecision]:
    """Parse and validate the LLM's JSON heading response."""

    json_text = _extract_json_payload(response_text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        preview = response_text[:1000]
        raise ValueError(
            "Could not parse JSON from LLM response. Raw response preview:\n"
            f"{preview}"
        ) from exc

    if isinstance(payload, list):
        raw_headings = payload
    elif isinstance(payload, dict):
        raw_headings = payload.get("section_headings", [])
    else:
        raise ValueError("LLM response must be a JSON object or list.")

    if not isinstance(raw_headings, list):
        raise ValueError("section_headings must be a list.")

    decisions: list[SectionHeadingDecision] = []
    seen_ids: set[int] = set()

    for item in raw_headings:
        if not isinstance(item, dict):
            continue

        candidate_id_value = item.get("candidate_id")

        if candidate_id_value is None:
            candidate_ids = item.get("candidate_ids", [])
            if isinstance(candidate_ids, list) and candidate_ids:
                candidate_id_value = candidate_ids[0]

        try:
            candidate_id = int(candidate_id_value)
        except (TypeError, ValueError):
            continue

        if candidate_id not in valid_candidate_ids:
            continue

        if candidate_id in seen_ids:
            continue

        title = item.get("section_title")

        if title is None or not str(title).strip():
            continue

        confidence = item.get("confidence")

        # Low-confidence selections are usually diagnostic output from the
        # model saying, in effect, "this might be something, but I am not sure."
        # For section boundary creation, it is safer to ignore them.
        if confidence is not None and str(confidence).strip().lower() == "low":
            continue

        number = item.get("section_number")

        decisions.append(
            SectionHeadingDecision(
                candidate_id=candidate_id,
                section_number=(
                    str(number).strip()
                    if number is not None and str(number).strip()
                    else None
                ),
                section_title=str(title).strip(),
                confidence=(
                    str(confidence).strip()
                    if confidence is not None and str(confidence).strip()
                    else None
                ),
            )
        )
        seen_ids.add(candidate_id)

    return decisions


def _build_singleton_json_prompt(candidate: HeadingCandidate) -> str:
    """Build a compact JSON prompt for judging one candidate."""

    prompt_dict = candidate_to_prompt_dict(candidate)
    example_heading = {
        "is_heading": True,
        "section_number": "1",
        "section_title": "Introduction",
        "confidence": "high",
    }
    example_not_heading = {
        "is_heading": False,
    }

    return (
        "You are helping kurrent identify section headings in an academic PDF.\n"
        "Decide whether this ONE candidate is an actual section heading.\n\n"
        "Reject page numbers, running headers/footers, chart axis labels, "
        "legend labels, figure/table labels, figure references, equation "
        "references, author lines, affiliations, and ordinary body prose.\n\n"
        "candidate_text is a deterministic best-effort cleaned heading prefix. "
        "raw_line is the original PDF-extracted line. If raw_line contains "
        "body text glued after the heading, extract only the heading itself.\n\n"
        "Return ONLY valid JSON. If this is a real section heading, use this "
        "shape:\n"
        f"{json.dumps(example_heading, ensure_ascii=False, indent=2)}\n\n"
        "If this is not a real section heading, use this shape:\n"
        f"{json.dumps(example_not_heading, ensure_ascii=False, indent=2)}\n\n"
        "Candidate:\n"
        f"{json.dumps(prompt_dict, ensure_ascii=False, indent=2)}"
    )


def _parse_singleton_heading_response(
    response_text: str,
    candidate: HeadingCandidate,
) -> list[SectionHeadingDecision]:
    """Parse and validate one-candidate JSON heading response."""

    json_text = _extract_json_payload(response_text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        preview = response_text[:1000]
        raise ValueError(
            "Could not parse JSON from singleton LLM response. "
            "Raw response preview:\n"
            f"{preview}"
        ) from exc

    # Be forgiving if the model accidentally returns the old batch shape.
    if isinstance(payload, dict) and "section_headings" in payload:
        return _parse_section_heading_response(
            response_text,
            valid_candidate_ids={candidate.candidate_id},
        )

    if not isinstance(payload, dict):
        raise ValueError("Singleton LLM response must be a JSON object.")

    is_heading = payload.get("is_heading")

    if isinstance(is_heading, str):
        is_heading = is_heading.strip().lower() in {"true", "yes", "y"}

    if not is_heading:
        return []

    confidence = payload.get("confidence")

    if confidence is not None and str(confidence).strip().lower() == "low":
        return []

    section_number_value = payload.get("section_number")
    section_title_value = payload.get("section_title")

    section_number = (
        str(section_number_value).strip()
        if section_number_value is not None and str(section_number_value).strip()
        else None
    )
    section_title = (
        str(section_title_value).strip()
        if section_title_value is not None and str(section_title_value).strip()
        else None
    )

    if section_title is None:
        candidate_text = _candidate_text_for_filtering(candidate)
        parsed_number, parsed_title = parse_section_heading(candidate_text)

        if section_number is None:
            section_number = parsed_number

        section_title = parsed_title or candidate_text

    return [
        SectionHeadingDecision(
            candidate_id=candidate.candidate_id,
            section_number=section_number,
            section_title=section_title,
            confidence=(
                str(confidence).strip()
                if confidence is not None and str(confidence).strip()
                else None
            ),
        )
    ]


def _chunked_candidates(
    candidates: list[HeadingCandidate],
    batch_size: int,
) -> list[list[HeadingCandidate]]:
    """Split candidates into small batches for local LLM calls."""

    if batch_size <= 0:
        return [candidates]

    return [
        candidates[i:i + batch_size]
        for i in range(0, len(candidates), batch_size)
    ]


def _sort_decisions_by_candidate_order(
    decisions: list[SectionHeadingDecision],
    candidates: list[HeadingCandidate],
) -> list[SectionHeadingDecision]:
    """Return decisions sorted by the candidate order used in the PDF."""

    order = {
        candidate.candidate_id: i
        for i, candidate in enumerate(candidates)
    }

    return sorted(
        decisions,
        key=lambda decision: order.get(decision.candidate_id, 10**9),
    )


def _dedupe_decisions(
    decisions: list[SectionHeadingDecision],
) -> list[SectionHeadingDecision]:
    """Remove duplicate decisions by candidate ID while preserving order."""

    seen: set[int] = set()
    deduped: list[SectionHeadingDecision] = []

    for decision in decisions:
        if decision.candidate_id in seen:
            continue

        seen.add(decision.candidate_id)
        deduped.append(decision)

    return deduped


def _ollama_chat(
    prompt: str,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT,
    use_json_format: bool = True,
) -> str:
    """Call Ollama's /api/chat endpoint and return response text."""

    url = ollama_url.rstrip("/") + "/api/chat"

    if use_json_format:
        system_content = (
            "You extract structured information from academic PDFs. "
            "You return only valid JSON and never invent candidate IDs."
        )
    else:
        system_content = (
            "You classify academic PDF heading candidates. "
            "Follow the requested output format exactly."
        )

    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    if use_json_format:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise OllamaTimeoutError(
            "Ollama timed out while selecting section headings."
        ) from exc
    except socket.timeout as exc:
        raise OllamaTimeoutError(
            "Ollama timed out while selecting section headings."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc

    try:
        content = response_payload["message"]["content"]
    except KeyError as exc:
        raise ValueError(
            f"Unexpected Ollama response payload: {response_payload!r}"
        ) from exc

    if content is None:
        raise ValueError(
            f"Ollama returned no message content: {response_payload!r}"
        )

    return str(content)


def _save_ollama_timeout_debug_file(
    prompt: str,
    candidates: list[HeadingCandidate],
    model: str,
    timeout_seconds: int,
    use_json_format: bool,
    fallback_kind: str,
) -> None:
    """Save a timeout-causing prompt for later reproduction/debugging."""

    debug_dir = Path(
        os.environ.get(
            "KURRENT_OLLAMA_DEBUG_DIR",
            str(Path.home() / ".kurrent" / "debug" / "ollama_timeouts"),
        )
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    id_text = "-".join(str(candidate_id) for candidate_id in candidate_ids)
    filename = f"ollama_timeout_{timestamp}_{fallback_kind}_{id_text}.json"

    payload = {
        "timestamp_utc": timestamp,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "use_json_format": use_json_format,
        "fallback_kind": fallback_kind,
        "candidate_ids": candidate_ids,
        "candidates": [candidate_to_prompt_dict(candidate) for candidate in candidates],
        "prompt": prompt,
    }

    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            "Warning: could not save Ollama timeout debug file: "
            f"{exc}",
            file=sys.stderr,
        )


def _build_singleton_yes_no_prompt(candidate: HeadingCandidate) -> str:
    """Build a simpler non-JSON fallback prompt for one candidate."""

    prompt_dict = candidate_to_prompt_dict(candidate)

    return (
        "You are helping identify real section headings in an academic PDF.\n"
        "Decide whether this one candidate is an actual section heading.\n"
        "Reject figure references, equation references, body prose, page "
        "headers/footers, author information, affiliations, and chart labels.\n"
        "Answer with exactly one word: YES or NO.\n\n"
        "Candidate:\n"
        f"{json.dumps(prompt_dict, ensure_ascii=False, indent=2)}"
    )


def _singleton_yes_no_fallback(
    candidate: HeadingCandidate,
    model: str,
    ollama_url: str,
    temperature: float,
    timeout_seconds: int,
) -> list[SectionHeadingDecision]:
    """Try a non-JSON YES/NO fallback for a single candidate."""

    prompt = _build_singleton_yes_no_prompt(candidate)

    try:
        response_text = _ollama_chat(
            prompt=prompt,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            num_predict=16,
            use_json_format=False,
        )
    except OllamaTimeoutError:
        _save_ollama_timeout_debug_file(
            prompt=prompt,
            candidates=[candidate],
            model=model,
            timeout_seconds=timeout_seconds,
            use_json_format=False,
            fallback_kind="singleton_yes_no",
        )
        return []

    normalized = response_text.strip().upper()

    if not normalized.startswith("YES"):
        return []

    candidate_text = _candidate_text_for_filtering(candidate)
    section_number, section_title = parse_section_heading(candidate_text)

    if section_title is None:
        section_title = candidate_text

    return [
        SectionHeadingDecision(
            candidate_id=candidate.candidate_id,
            section_number=section_number,
            section_title=section_title,
            confidence="fallback",
        )
    ]


def _select_heading_batch_with_timeout_fallback(
    candidates: list[HeadingCandidate],
    model: str,
    ollama_url: str,
    temperature: float,
    timeout_seconds: int,
    singleton_timeout_seconds: int,
    num_predict: int,
    depth: int = 0,
) -> list[SectionHeadingDecision]:
    """Select headings, recursively splitting a failed batch.

    The default batch size is now one candidate, which makes each prompt and
    each expected JSON response very small. This function still supports larger
    experimental batches via KURRENT_OLLAMA_SECTION_BATCH_SIZE. If a batch
    times out or returns malformed/truncated JSON, the same payload is retried
    once. If it fails again, the batch is bisected. A failing singleton falls
    back to a simple non-JSON YES/NO prompt and is skipped if that also fails.
    """

    if not candidates:
        return []

    effective_timeout = timeout_seconds

    if len(candidates) == 1:
        effective_timeout = min(timeout_seconds, singleton_timeout_seconds)
        prompt = _build_singleton_json_prompt(candidates[0])
        prompt_kind = "singleton_json"
        effective_num_predict = min(num_predict, 128)
    else:
        prompt = _build_section_heading_prompt(candidates)
        prompt_kind = "batch_json"
        effective_num_predict = num_predict

    last_failure: str | None = None

    for attempt_number in range(1, 3):
        try:
            response_text = _ollama_chat(
                prompt=prompt,
                model=model,
                ollama_url=ollama_url,
                temperature=temperature,
                timeout_seconds=effective_timeout,
                num_predict=effective_num_predict,
                use_json_format=True,
            )

            if len(candidates) == 1:
                return _parse_singleton_heading_response(
                    response_text,
                    candidate=candidates[0],
                )

            return _parse_section_heading_response(
                response_text,
                valid_candidate_ids={
                    candidate.candidate_id
                    for candidate in candidates
                },
            )
        except OllamaTimeoutError:
            last_failure = "timeout"
            _save_ollama_timeout_debug_file(
                prompt=prompt,
                candidates=candidates,
                model=model,
                timeout_seconds=effective_timeout,
                use_json_format=True,
                fallback_kind=f"{prompt_kind}_timeout_attempt_{attempt_number}",
            )
        except ValueError as exc:
            last_failure = "malformed JSON"
            _save_ollama_timeout_debug_file(
                prompt=prompt,
                candidates=candidates,
                model=model,
                timeout_seconds=effective_timeout,
                use_json_format=True,
                fallback_kind=f"{prompt_kind}_parse_attempt_{attempt_number}",
            )

            if attempt_number == 2:
                print(
                    "Warning: Ollama returned malformed/truncated JSON while "
                    "selecting section headings for "
                    f"{len(candidates)} candidate(s): {exc}",
                    file=sys.stderr,
                )

        if attempt_number == 1:
            print(
                "Warning: Ollama failed while selecting section headings "
                f"for {len(candidates)} candidate(s) ({last_failure}); "
                "retrying the same payload once.",
                file=sys.stderr,
            )
            continue

    if len(candidates) == 1:
        candidate = candidates[0]

        print(
            "Warning: Ollama failed twice on a single heading candidate; "
            "trying simpler non-JSON YES/NO fallback for "
            f"candidate_id={candidate.candidate_id} on page {candidate.page}: "
            f"{_candidate_text_for_filtering(candidate)!r}",
            file=sys.stderr,
        )

        yes_no_decisions = _singleton_yes_no_fallback(
            candidate=candidate,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature,
            timeout_seconds=effective_timeout,
        )

        if yes_no_decisions:
            return yes_no_decisions

        print(
            "Warning: skipping single heading candidate after LLM/fallback "
            f"failure: candidate_id={candidate.candidate_id} "
            f"on page {candidate.page}: "
            f"{_candidate_text_for_filtering(candidate)!r}",
            file=sys.stderr,
        )
        raise _HeadingCandidateFailedError(
            "Ollama could not classify heading candidate "
            f"candidate_id={candidate.candidate_id} on page {candidate.page}."
        )

    midpoint = len(candidates) // 2
    left = candidates[:midpoint]
    right = candidates[midpoint:]

    print(
        "Warning: Ollama failed twice while selecting section headings "
        f"for {len(candidates)} candidates ({last_failure}); retrying as "
        f"{len(left)} + {len(right)} candidates.",
        file=sys.stderr,
    )

    return (
        _select_heading_batch_with_timeout_fallback(
            candidates=left,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            singleton_timeout_seconds=singleton_timeout_seconds,
            num_predict=num_predict,
            depth=depth + 1,
        )
        + _select_heading_batch_with_timeout_fallback(
            candidates=right,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            singleton_timeout_seconds=singleton_timeout_seconds,
            num_predict=num_predict,
            depth=depth + 1,
        )
    )


def _drop_reference_entry_decisions(
    decisions: list[SectionHeadingDecision],
    candidates: list[HeadingCandidate],
) -> list[SectionHeadingDecision]:
    """Drop accepted decisions that are actually bibliography entries.

    The deterministic pre-filter should prevent these from reaching Ollama, but
    this post-filter provides a second guard. It is especially useful once a
    real References heading has been accepted: any later "section" that looks
    like a numbered bibliography entry should not create a new SectionSpan.
    """

    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }
    kept: list[SectionHeadingDecision] = []
    in_references = False

    for decision in decisions:
        title = _normalize_for_match(decision.section_title)

        if title in {"references", "bibliography", "works cited", "literature cited"}:
            in_references = True
            kept.append(decision)
            continue

        candidate = candidate_by_id.get(decision.candidate_id)

        if candidate is not None:
            if _looks_like_reference_entry_candidate(candidate):
                continue

            if in_references:
                candidate_text = _candidate_text_for_filtering(candidate)
                raw_line = _normalize_spaces(candidate.line_text)

                if (
                    _looks_like_numbered_reference_entry(candidate_text)
                    or _looks_like_numbered_reference_entry(raw_line)
                    or _looks_like_unnumbered_reference_entry_fragment(candidate_text)
                ):
                    continue

        kept.append(decision)

    return kept

def select_section_headings_with_ollama(
    candidates: list[HeadingCandidate],
    model: str | None = None,
    ollama_url: str | None = None,
    temperature: float = 0.0,
    batch_size: int | None = None,
    timeout_seconds: int | None = None,
    singleton_timeout_seconds: int | None = None,
    num_predict: int | None = None,
    max_consecutive_failures: int | None = None,
    progress_total_callback: Callable[[int], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[SectionHeadingDecision]:
    """Ask Ollama to select real section headings from candidates.

    Candidates are sent one at a time by default. This keeps local Ollama
    prompts and JSON responses small, which is usually more reliable than
    asking for one large JSON list of section-heading decisions.

    If progress_total_callback is provided, it receives the number of filtered
    candidates that will be sent to Ollama. If progress_callback is provided,
    it receives the number of candidates completed after each Ollama batch.
    With the default batch size of 1, this means one progress update per
    candidate.
    """

    candidates = filtered_candidates(candidates)

    if progress_total_callback is not None:
        progress_total_callback(len(candidates))

    if not candidates:
        return []

    model = model or DEFAULT_OLLAMA_MODEL
    ollama_url = ollama_url or DEFAULT_OLLAMA_URL

    if batch_size is None:
        batch_size = int(
            os.environ.get(
                "KURRENT_OLLAMA_SECTION_BATCH_SIZE",
                DEFAULT_OLLAMA_BATCH_SIZE,
            )
        )

    if timeout_seconds is None:
        timeout_seconds = int(
            os.environ.get(
                "KURRENT_OLLAMA_TIMEOUT_SECONDS",
                DEFAULT_OLLAMA_TIMEOUT_SECONDS,
            )
        )

    if singleton_timeout_seconds is None:
        singleton_timeout_seconds = int(
            os.environ.get(
                "KURRENT_OLLAMA_SINGLETON_TIMEOUT_SECONDS",
                DEFAULT_OLLAMA_SINGLETON_TIMEOUT_SECONDS,
            )
        )

    if num_predict is None:
        num_predict = int(
            os.environ.get(
                "KURRENT_OLLAMA_NUM_PREDICT",
                DEFAULT_OLLAMA_NUM_PREDICT,
            )
        )

    if max_consecutive_failures is None:
        max_consecutive_failures = int(
            os.environ.get(
                "KURRENT_OLLAMA_MAX_CONSECUTIVE_SECTION_FAILURES",
                DEFAULT_OLLAMA_MAX_CONSECUTIVE_FAILURES,
            )
        )

    decisions: list[SectionHeadingDecision] = []
    consecutive_failures = 0

    for batch in _chunked_candidates(candidates, batch_size):
        try:
            batch_decisions = _select_heading_batch_with_timeout_fallback(
                candidates=batch,
                model=model,
                ollama_url=ollama_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                singleton_timeout_seconds=singleton_timeout_seconds,
                num_predict=num_predict,
            )
        except _HeadingCandidateFailedError as exc:
            consecutive_failures += 1

            if progress_callback is not None:
                progress_callback(len(batch))

            if max_consecutive_failures <= 0 or consecutive_failures < max_consecutive_failures:
                continue

            raise LLMSectioningUnavailableError(
                "Ollama failed on "
                f"{consecutive_failures} consecutive section-heading candidate(s); "
                "falling back to rules-based sectioning for this document."
            ) from exc
        except RuntimeError as exc:
            raise LLMSectioningUnavailableError(
                "Ollama was unavailable while selecting section headings; "
                "falling back to rules-based sectioning for this document."
            ) from exc

        consecutive_failures = 0
        decisions.extend(batch_decisions)

        if progress_callback is not None:
            progress_callback(len(batch))

    decisions = _dedupe_decisions(decisions)
    decisions = _drop_reference_entry_decisions(decisions, candidates)

    return _sort_decisions_by_candidate_order(decisions, candidates)
