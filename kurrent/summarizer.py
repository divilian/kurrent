"""Screening summaries for PDFs before they are ingested into kurrent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from kurrent.cli_display import collapse_whitespace
from kurrent.config import (
    DEFAULT_OLLAMA_SUMMARY_MAX_NUM_CTX,
    DEFAULT_OLLAMA_SUMMARY_MODEL,
    DEFAULT_OLLAMA_SUMMARY_TIMEOUT,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SUMMARY_DEPTH,
)
from kurrent.schema import SectionSpan
from kurrent.sectioner import (
    detect_heading_candidates,
    is_reference_section_title,
    make_section_spans_from_headings,
    make_section_spans_from_llm_decisions,
    normalize_section_title,
)

__all__ = [
    "DEFAULT_SCREENING_SUMMARY_OUTPUT_TOKENS_PER_DEPTH_UNIT",
    "DEFAULT_SCREENING_SUMMARY_MAX_SECTIONS",
    "ScreeningSummaryError",
    "ScreeningSectionExcerpt",
    "ScreeningSummary",
    "estimate_tokens",
    "trim_section_for_summary",
    "is_obvious_non_content_section",
    "section_display_title",
    "screening_sections_for_pdf",
    "select_screening_sections",
    "clean_screening_summary_text",
    "summarize_pdf_for_screening",
]

DEFAULT_SCREENING_SUMMARY_OUTPUT_TOKENS_PER_DEPTH_UNIT = 190
DEFAULT_SCREENING_SUMMARY_MAX_SECTIONS = 5
TRUNCATION_MARKER = (
    "\n\n[Middle of this unusually long section omitted for the screening "
    "summary.]\n\n"
)
COMBINED_SECTION_SEPARATOR = "\n\n---\n\n"


class ScreeningSummaryError(RuntimeError):
    """Raised when a screening summary cannot be generated."""


@dataclass(frozen=True, slots=True)
class ScreeningSectionExcerpt:
    """One selected section excerpt used for a fast screening summary."""

    section_title: str
    text: str
    was_truncated: bool = False


@dataclass(frozen=True, slots=True)
class ScreeningSummary:
    """The final screening summary and the section excerpts used to create it."""

    text: str
    section_notes: tuple[ScreeningSectionExcerpt, ...]
    depth: int


# Backward-compatible name for older tests/imports from the first implementation.
SectionSummaryNote = ScreeningSectionExcerpt


def estimate_tokens(text: str) -> int:
    """Return a rough, dependency-free token estimate for prompt budgeting."""

    if not text:
        return 0

    # Four characters per token is intentionally approximate. The goal is not
    # exact tokenizer parity; it is a conservative guardrail before local
    # Ollama calls.
    return max(1, math.ceil(len(text) / 4))


def trim_section_for_summary(
    text: str,
    *,
    max_num_ctx: int = DEFAULT_OLLAMA_SUMMARY_MAX_NUM_CTX,
) -> tuple[str, bool]:
    """Return section text that fits the summary prompt budget.

    max_num_ctx is an Ollama-style context budget. Kurrent uses a rough
    character estimate to avoid adding a tokenizer dependency here. If the
    section is too large, preserve the beginning and end with an explicit
    omission marker between them.
    """

    text = collapse_whitespace(text)

    if estimate_tokens(text) <= max_num_ctx:
        return text, False

    max_chars = max(1_000, max_num_ctx * 4)
    marker = TRUNCATION_MARKER.strip()
    available_chars = max(1_000, max_chars - len(marker) - 4)
    tail_chars = max(250, available_chars // 3)
    head_chars = max(250, available_chars - tail_chars)

    trimmed = (
        text[:head_chars].rstrip()
        + TRUNCATION_MARKER
        + text[-tail_chars:].lstrip()
    )
    return trimmed, True


def is_obvious_non_content_section(section: SectionSpan) -> bool:
    """Return whether a section should be skipped for screening summaries."""

    title = section.section_title

    if title is None:
        return False

    normalized = normalize_section_title(title)

    if is_reference_section_title(title):
        return True

    return normalized in {
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "funding",
        "author contributions",
        "competing interests",
        "conflict of interest",
        "conflicts of interest",
    }


def section_display_title(section: SectionSpan) -> str:
    """Return a user-facing section label for progress messages."""

    if section.section_title and section.section_title.strip():
        return section.section_title.strip()

    if section.section_number and section.section_number.strip():
        return f"Section {section.section_number.strip()}"

    if section.section_index is None:
        return "front matter"

    return f"section {section.section_index + 1}"


def screening_sections_for_pdf(
    pdf_path: str | Path,
    *,
    llm_sectioning_prefetch=None,
) -> list[SectionSpan]:
    """Return the best immediately available sections for a screening summary.

    If background LLM section recognition has already finished successfully,
    use it. Otherwise, do not wait: fall back to rules-based section detection.
    """

    if (
        llm_sectioning_prefetch is not None
        and getattr(llm_sectioning_prefetch, "unavailable_error", None) is None
    ):
        candidates = getattr(llm_sectioning_prefetch, "candidates", None)
        decisions = getattr(llm_sectioning_prefetch, "decisions", None)
        if candidates is not None and decisions is not None:
            return make_section_spans_from_llm_decisions(
                pdf_path=pdf_path,
                doc_id="screening-summary",
                candidates=candidates,
                decisions=decisions,
            )

    headings = detect_heading_candidates(pdf_path)
    return make_section_spans_from_headings(
        pdf_path=pdf_path,
        doc_id="screening-summary",
        headings=headings,
    )


def _call_ollama_summary_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    ollama_url: str,
    timeout_seconds: float,
    num_ctx: int,
    num_predict: int,
) -> str:
    """Call Ollama's chat API for summarization."""

    api_url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ScreeningSummaryError(
            f"Ollama summary unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    content = response_data.get("message", {}).get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise ScreeningSummaryError("Ollama returned an empty summary response.")

    return content.strip()


def _normalized_text_fingerprint(text: str) -> str:
    normalized = collapse_whitespace(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _section_priority(section: SectionSpan) -> tuple[int, int]:
    """Return a heuristic priority without imposing a document genre schema."""

    title = normalize_section_title(section.section_title or "")
    index = section.section_index if section.section_index is not None else -1

    if title in {"abstract", "summary", "executive summary"}:
        return (0, index)
    if title in {"introduction", "intro", "overview", "background"}:
        return (1, index)
    if re.search(r"\b(conclusion|conclusions|discussion|future work)\b", title):
        return (2, index)
    if title in {"front matter", "title page"} or index < 0:
        return (4, index)
    return (3, index)


def _section_order(section: SectionSpan) -> int:
    return section.section_index if section.section_index is not None else -1


def select_screening_sections(
    sections: list[SectionSpan],
    *,
    max_sections: int = DEFAULT_SCREENING_SUMMARY_MAX_SECTIONS,
    max_num_ctx: int = DEFAULT_OLLAMA_SUMMARY_MAX_NUM_CTX,
) -> list[ScreeningSectionExcerpt]:
    """Select a small, representative set of section excerpts for one LLM call.

    The selector stays genre-agnostic: it does not require methods/results/etc.
    It merely prefers common high-signal sections when present, otherwise it
    falls back to early content sections. Duplicate titles and exact duplicate
    text are suppressed because mixed rules/LLM section sources can overlap.
    """

    content_sections = [
        section
        for section in sections
        if section.text.strip() and not is_obvious_non_content_section(section)
    ]

    if not content_sections:
        return []

    # Prefer the first occurrence of repeated section titles and exact text. In
    # practice, repeated high-level/nested section passes can otherwise double
    # count the same conceptual section during ingest screening.
    deduped: list[SectionSpan] = []
    seen_titles: set[str] = set()
    seen_texts: set[str] = set()
    for section in content_sections:
        title_key = normalize_section_title(section_display_title(section))
        text_key = _normalized_text_fingerprint(section.text)
        if title_key in seen_titles or text_key in seen_texts:
            continue
        seen_titles.add(title_key)
        seen_texts.add(text_key)
        deduped.append(section)

    ranked = sorted(deduped, key=_section_priority)
    selected = sorted(ranked[:max_sections], key=_section_order)

    prompt_budget = max(250, max_num_ctx - 1_500)
    per_section_budget = max(100, prompt_budget // max(1, len(selected)))

    excerpts: list[ScreeningSectionExcerpt] = []
    used_tokens = 0
    for section in selected:
        if used_tokens >= prompt_budget:
            break
        remaining = max(100, prompt_budget - used_tokens)
        section_budget = min(per_section_budget, remaining)
        text, was_truncated = trim_section_for_summary(
            section.text,
            max_num_ctx=section_budget,
        )
        title = section_display_title(section)
        section_tokens = estimate_tokens(title) + estimate_tokens(text) + 20
        if used_tokens + section_tokens > prompt_budget and excerpts:
            break
        excerpts.append(
            ScreeningSectionExcerpt(
                section_title=title,
                text=text,
                was_truncated=was_truncated,
            )
        )
        used_tokens += section_tokens

    return excerpts


def _screening_summary_messages(
    excerpts: list[ScreeningSectionExcerpt],
    *,
    depth: int,
) -> list[dict[str, str]]:
    system = (
        "You write fast screening summaries to help a human decide whether to "
        "ingest a PDF into a research library. Be accurate, concrete, and "
        "genre-agnostic. Do not assume this is an empirical paper. Do not invent "
        "methods, results, datasets, conclusions, or relevance to the user's "
        "research interests."
    )
    section_text = COMBINED_SECTION_SEPARATOR.join(
        f"## {excerpt.section_title}\n{excerpt.text}"
        for excerpt in excerpts
    )
    user = f"""
Write a concise screening summary to help a human decide whether to ingest this PDF into a research library.

Aim for about {depth} short, modular prose paragraph(s). Each paragraph should do one clear job, such as explaining what the document is about, what approach/model/evidence/argument it develops, or what it appears to contribute. Avoid one oversized run-on paragraph when the selected text supports more than one coherent summary unit. The requested number is a target for summary depth, not a rigid quota: do not pad the summary or invent artificial structure if the selected text supports fewer coherent paragraphs.

Use readable plain text. Do not use Markdown headings, bold text, or labels such as "Major Point 1". A concise numbered or bulleted list is acceptable only when the document genuinely presents distinct contributions, findings, components, or steps and the list makes the summary clearer. Do not split one idea merely to satisfy the requested depth.

Focus on what the document appears to be about, what kind of work it is, and what it appears to contribute or argue. Do not make claims about whether it is relevant to Kurrent, to the user's research interests, to any existing library/corpus, or to generic groups such as "researchers interested in" a topic unless the document itself makes that claim. Do not mention Kurrent. Do not include meta-commentary such as "This summary indicates," "Overall," or "In summary"; just provide the screening summary itself.

The selected section text may be incomplete or truncated. Summarize only what is present and do not infer missing details.

<selected sections>
{section_text}
</selected sections>
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _clean_markdown_symbols(line: str) -> str:
    """Strip common Markdown styling without restructuring the summary.

    Preserve ordinary list markers because a short list can expose real
    document structure. Remove only decoration that makes an ingest-screening
    summary look noisy, such as Markdown headings, bold emphasis, and inline
    code ticks.
    """

    line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"__([^_]+)__", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    return line.strip()


def clean_screening_summary_text(text: str, depth: int | None = None) -> str:
    """Return lightly cleaned summary text without forcing summary depth.

    The requested summary-depth is a target for substantive depth, not a
    formatting rule. Avoid mechanical paragraph splitting/merging here, but
    remove obvious Markdown symbols that make screening summaries look noisy.
    """

    del depth

    paragraphs: list[str] = []
    for part in re.split(r"\n\s*\n", text.strip()):
        cleaned_lines = [
            _clean_markdown_symbols(line)
            for line in part.splitlines()
            if line.strip()
        ]
        cleaned_part = "\n".join(line for line in cleaned_lines if line).strip()
        if cleaned_part:
            paragraphs.append(cleaned_part)

    return "\n\n".join(paragraphs)


def _normalize_depth(depth: int | None) -> int:
    if depth is None:
        return DEFAULT_SUMMARY_DEPTH

    return max(1, int(depth))


def summarize_pdf_for_screening(
    pdf_path: str | Path,
    *,
    depth: int | None = None,
    model: str = DEFAULT_OLLAMA_SUMMARY_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = DEFAULT_OLLAMA_SUMMARY_TIMEOUT,
    max_num_ctx: int = DEFAULT_OLLAMA_SUMMARY_MAX_NUM_CTX,
    llm_sectioning_prefetch=None,
    progress_callback: Callable[[str], None] | None = None,
    answer_function: Callable[[list[dict[str, str]]], str] | None = None,
) -> ScreeningSummary:
    """Generate a fast one-call screening summary for one PDF."""

    depth = _normalize_depth(depth)

    if progress_callback is not None:
        progress_callback("Selecting sections for screening summary...")

    sections = screening_sections_for_pdf(
        pdf_path,
        llm_sectioning_prefetch=llm_sectioning_prefetch,
    )
    selected_sections = select_screening_sections(
        sections,
        max_num_ctx=max_num_ctx,
    )

    if not selected_sections:
        raise ScreeningSummaryError("No extractable content sections found to summarize.")

    if progress_callback is not None:
        names = ", ".join(excerpt.section_title for excerpt in selected_sections)
        progress_callback(f"Summarizing selected sections: {names}...")

    messages = _screening_summary_messages(
        selected_sections,
        depth=depth,
    )

    if answer_function is not None:
        summary_text = answer_function(messages)
    else:
        summary_text = _call_ollama_summary_chat(
            messages,
            model=model,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
            num_ctx=max_num_ctx,
            num_predict=max(
                180,
                DEFAULT_SCREENING_SUMMARY_OUTPUT_TOKENS_PER_DEPTH_UNIT
                * depth,
            ),
        )

    return ScreeningSummary(
        text=clean_screening_summary_text(summary_text, depth),
        section_notes=tuple(selected_sections),
        depth=depth,
    )
