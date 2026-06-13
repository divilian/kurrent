"""Create temporary highlighted PDF copies for Kurrent source navigation.

The highlighter is intentionally best-effort. It never mutates the user's
corpus PDF. Instead it tries to locate a query-relevant excerpt on a source page,
highlights the matching words in a temporary copy, and reports a graceful failure
when the page text cannot be matched robustly.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from kurrent.cli_display import collapse_whitespace
from kurrent.config import DEFAULT_OLLAMA_URL, DEFAULT_PDF_EXCERPT_LLM

DEFAULT_OLLAMA_MODEL = DEFAULT_PDF_EXCERPT_LLM
DEFAULT_HIGHLIGHT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_PAGE_TEXT_CHARS = 12_000
DEFAULT_MIN_MATCH_SCORE = 0.72
DEFAULT_MIN_EXCERPT_WORDS = 12
DEFAULT_MAX_EXCERPT_WORDS = 120
DEFAULT_MAX_HIGHLIGHT_PAGE_RANGE = 4


@dataclass(frozen=True, slots=True)
class HighlightResult:
    """Result of trying to create a temporary highlighted PDF."""

    original_pdf_path: Path
    highlighted_pdf_path: Path | None
    page: int | None
    success: bool
    matched_excerpt: str | None = None
    method: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PageWord:
    """One PDF word with its rendered coordinates and normalized form."""

    text: str
    normalized: str
    rect: object
    block_no: int
    line_no: int


@dataclass(frozen=True, slots=True)
class WordSpanMatch:
    """A matched span of PDF words."""

    start: int
    end: int
    score: float
    method: str


ExcerptSelector = Callable[[str, str], str | None]


LIGATURE_TRANSLATION = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)


def normalize_match_token(text: str) -> str:
    """Normalize one word for fuzzy PDF-text matching."""

    text = text.translate(LIGATURE_TRANSLATION)
    text = text.replace("\u00ad", "")
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalized_tokens(text: str) -> list[str]:
    """Return normalized tokens for matching, dropping punctuation-only words."""

    tokens = []

    for raw_token in collapse_whitespace(text).split():
        token = normalize_match_token(raw_token)

        if token:
            tokens.append(token)

    return tokens


def _valid_page_number(page: int | None) -> int | None:
    if page is None:
        return None

    try:
        page = int(page)
    except (TypeError, ValueError):
        return None

    if page < 1:
        return None

    return page


def _page_words(page) -> list[PageWord]:
    """Return normalized page words with coordinates from a PyMuPDF page."""

    import fitz

    words = []

    # x0, y0, x1, y1, word, block_no, line_no, word_no
    for item in page.get_text("words"):
        if len(item) < 8:
            continue

        x0, y0, x1, y1, text, block_no, line_no, _word_no = item[:8]
        normalized = normalize_match_token(str(text))

        if not normalized:
            continue

        words.append(
            PageWord(
                text=str(text),
                normalized=normalized,
                rect=fitz.Rect(x0, y0, x1, y1),
                block_no=int(block_no),
                line_no=int(line_no),
            )
        )

    return words


def clean_page_text_from_words(words: Iterable[PageWord]) -> str:
    """Return readable page text for the LLM excerpt selector."""

    return collapse_whitespace(" ".join(word.text for word in words))


def _truncate_word_count(text: str, min_words: int, max_words: int) -> str:
    """Keep an excerpt at a locate-able but visually manageable length."""

    words = collapse_whitespace(text).split()

    if len(words) <= max_words:
        return " ".join(words)

    if max_words < min_words:
        max_words = min_words

    return " ".join(words[:max_words])


def _ollama_excerpt_messages(
    page_text: str,
    research_interest: str,
    evidence_excerpt: str | None = None,
) -> list[dict[str, str]]:
    """Build messages for selecting a short page excerpt to highlight."""

    page_text = collapse_whitespace(page_text)
    research_interest = collapse_whitespace(research_interest)
    evidence_excerpt = collapse_whitespace(evidence_excerpt or "")

    if len(page_text) > DEFAULT_MAX_PAGE_TEXT_CHARS:
        page_text = page_text[:DEFAULT_MAX_PAGE_TEXT_CHARS].rstrip() + " [...]"

    evidence_block = ""
    evidence_instruction = (
        "Select the shortest contiguous excerpt from the page text that is most "
        "relevant to the research interest."
    )

    if evidence_excerpt:
        if len(evidence_excerpt) > DEFAULT_MAX_PAGE_TEXT_CHARS:
            evidence_excerpt = evidence_excerpt[:DEFAULT_MAX_PAGE_TEXT_CHARS].rstrip() + " [...]"

        evidence_block = f"""
Retrieved evidence excerpt for this specific source item:
{evidence_excerpt}
""".strip()
        evidence_instruction = (
            "Select the shortest contiguous excerpt from the page text that best "
            "corresponds to the retrieved evidence excerpt for this specific source "
            "item, while still being relevant to the research interest."
        )

    system_message = (
        "You select exact text spans from academic PDF pages. Return only JSON. "
        "Do not paraphrase."
    )
    user_parts = [
        f"Research interest:\n{research_interest}",
    ]

    if evidence_block:
        user_parts.append(evidence_block)

    user_parts.append(f"Page text:\n{page_text}")
    user_parts.append(
        f"""
Task:
{evidence_instruction} The excerpt must be copied verbatim from the page text,
except that whitespace may be collapsed. Prefer 25-120 words. Do not select page
headers, journal labels, title text, bylines, or boilerplate unless they are the
only text that corresponds to the evidence. If no passage is meaningfully
relevant, return an empty excerpt.

Return exactly this JSON shape:
{{"excerpt": "..."}}
""".strip()
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def select_relevant_excerpt_with_ollama(
    page_text: str,
    research_interest: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = DEFAULT_HIGHLIGHT_TIMEOUT_SECONDS,
    evidence_excerpt: str | None = None,
) -> str | None:
    """Ask Ollama for an exact relevant excerpt copied from page_text."""

    page_text = collapse_whitespace(page_text)
    research_interest = collapse_whitespace(research_interest)

    if not page_text or not research_interest:
        return None

    payload = {
        "model": model,
        "messages": _ollama_excerpt_messages(
            page_text,
            research_interest,
            evidence_excerpt=evidence_excerpt,
        ),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 300},
    }
    api_url = f"{ollama_url.rstrip('/')}/api/chat"
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    content = response_data.get("message", {}).get("content", "")

    if not isinstance(content, str) or not content.strip():
        return None

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    excerpt = data.get("excerpt")

    if not isinstance(excerpt, str) or not excerpt.strip():
        return None

    return _truncate_word_count(
        excerpt,
        min_words=DEFAULT_MIN_EXCERPT_WORDS,
        max_words=DEFAULT_MAX_EXCERPT_WORDS,
    )

def _exact_token_match(page_tokens: list[str], excerpt_tokens: list[str]) -> WordSpanMatch | None:
    """Return an exact contiguous token match, if available."""

    if not page_tokens or not excerpt_tokens:
        return None

    excerpt_len = len(excerpt_tokens)

    if excerpt_len > len(page_tokens):
        return None

    for start in range(0, len(page_tokens) - excerpt_len + 1):
        end = start + excerpt_len

        if page_tokens[start:end] == excerpt_tokens:
            return WordSpanMatch(start=start, end=end, score=1.0, method="exact-token")

    return None


def _window_similarity(a: list[str], b: list[str]) -> float:
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def fuzzy_match_excerpt_to_words(
    page_words: list[PageWord],
    excerpt: str,
    min_score: float = DEFAULT_MIN_MATCH_SCORE,
) -> WordSpanMatch | None:
    """Find the page-word span that best matches an excerpt."""

    page_tokens = [word.normalized for word in page_words]
    excerpt_tokens = normalized_tokens(excerpt)

    if not page_tokens or not excerpt_tokens:
        return None

    exact = _exact_token_match(page_tokens, excerpt_tokens)

    if exact is not None:
        return exact

    excerpt_len = len(excerpt_tokens)
    best: WordSpanMatch | None = None
    min_window = max(1, int(excerpt_len * 0.75))
    max_window = min(len(page_tokens), max(min_window, int(excerpt_len * 1.35) + 2))

    for window_len in range(min_window, max_window + 1):
        for start in range(0, len(page_tokens) - window_len + 1):
            end = start + window_len
            score = _window_similarity(excerpt_tokens, page_tokens[start:end])

            if best is None or score > best.score:
                best = WordSpanMatch(
                    start=start,
                    end=end,
                    score=score,
                    method="fuzzy-token",
                )

    if best is None or best.score < min_score:
        return None

    return best



def _fallback_excerpt_is_safe_direct_match(excerpt: str | None) -> bool:
    """Return whether fallback text is short enough to match directly.

    Evidence packets are often whole chunks. Directly matching a long chunk tends
    to highlight the beginning of the chunk, including page headers and titles,
    rather than the specific paragraph the user selected. Long evidence should be
    used as context for the selector, not highlighted wholesale.
    """

    if not excerpt:
        return False

    token_count = len(normalized_tokens(excerpt))
    return 0 < token_count <= DEFAULT_MAX_EXCERPT_WORDS


def _candidate_pages(page_start: int, page_end: int | None, page_count: int) -> list[int]:
    """Return 1-based candidate pages to inspect for a source passage."""

    if page_end is None or page_end < page_start:
        page_end = page_start

    page_start = max(1, page_start)
    page_end = min(page_end, page_count)

    if page_start > page_end:
        return []

    if page_end - page_start + 1 > DEFAULT_MAX_HIGHLIGHT_PAGE_RANGE:
        page_end = page_start + DEFAULT_MAX_HIGHLIGHT_PAGE_RANGE - 1

    return list(range(page_start, page_end + 1))


def _select_excerpt_for_page(
    page_text: str,
    research_interest: str,
    fallback_excerpt: str | None,
    excerpt_selector: ExcerptSelector | None,
    model: str,
    ollama_url: str,
    timeout_seconds: float,
) -> str | None:
    """Select a short excerpt to highlight on one page."""

    if excerpt_selector is not None:
        excerpt = excerpt_selector(page_text, research_interest)
    else:
        excerpt = select_relevant_excerpt_with_ollama(
            page_text,
            research_interest,
            model=model,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
            evidence_excerpt=fallback_excerpt,
        )

    if excerpt is not None and excerpt.strip():
        return excerpt

    if _fallback_excerpt_is_safe_direct_match(fallback_excerpt):
        return _truncate_word_count(
            fallback_excerpt or "",
            min_words=DEFAULT_MIN_EXCERPT_WORDS,
            max_words=DEFAULT_MAX_EXCERPT_WORDS,
        )

    return None

def _line_rects_for_match(page_words: list[PageWord], match: WordSpanMatch):
    """Return one union rectangle per matched visual line."""

    line_rects = []
    current_key = None
    current_rect = None

    for word in page_words[match.start : match.end]:
        key = (word.block_no, word.line_no)

        if key != current_key:
            if current_rect is not None:
                line_rects.append(current_rect)

            current_key = key
            current_rect = word.rect
            continue

        current_rect = current_rect | word.rect

    if current_rect is not None:
        line_rects.append(current_rect)

    return line_rects


def _highlight_rects(page, rects) -> None:
    """Add highlight annotations for line rectangles."""

    if not rects:
        return

    try:
        annot = page.add_highlight_annot(rects)

        if annot is not None:
            annot.update()
            return
    except Exception:
        pass

    for rect in rects:
        annot = page.add_highlight_annot(rect)

        if annot is not None:
            annot.update()


def _highlight_output_path(pdf_path: Path, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "kurrent" / "highlights"

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem[:80] or "source"
    return output_dir / f"{stem}-highlight-{uuid4().hex[:12]}.pdf"


def create_highlighted_pdf_for_research_interest(
    pdf_path: str | Path,
    page_start: int | None,
    research_interest: str,
    page_end: int | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = DEFAULT_HIGHLIGHT_TIMEOUT_SECONDS,
    fallback_excerpt: str | None = None,
    excerpt_selector: ExcerptSelector | None = None,
    output_dir: Path | None = None,
) -> HighlightResult:
    """Create a temporary highlighted PDF copy for a source page.

    The selector is asked to choose a query-relevant excerpt from the clean page
    text. Kurrent then maps that excerpt back to rendered PDF word coordinates
    and highlights the matching lines. If any step fails, success is False and
    callers should open the original PDF instead.
    """

    pdf_path = Path(pdf_path)
    page = _valid_page_number(page_start)

    if page is None:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=None,
            success=False,
            message="No usable page number is available for highlighting.",
        )

    if not pdf_path.exists():
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=page,
            success=False,
            message=f"PDF path does not exist: {pdf_path}",
        )

    try:
        import fitz
    except ImportError as exc:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=page,
            success=False,
            message=f"PyMuPDF is not available: {exc}",
        )

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=page,
            success=False,
            message=f"Could not open PDF for highlighting: {type(exc).__name__}: {exc}",
        )

    try:
        candidate_pages = _candidate_pages(page, page_end, document.page_count)

        if not candidate_pages:
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=page,
                success=False,
                message=f"Page {page} is outside the PDF page range.",
            )

        last_failure: HighlightResult | None = None

        for candidate_page in candidate_pages:
            pdf_page = document[candidate_page - 1]
            page_words = _page_words(pdf_page)

            if not page_words:
                last_failure = HighlightResult(
                    original_pdf_path=pdf_path,
                    highlighted_pdf_path=None,
                    page=candidate_page,
                    success=False,
                    message="No extractable words were found on this PDF page.",
                )
                continue

            page_text = clean_page_text_from_words(page_words)
            excerpt = _select_excerpt_for_page(
                page_text=page_text,
                research_interest=research_interest,
                fallback_excerpt=fallback_excerpt,
                excerpt_selector=excerpt_selector,
                model=model,
                ollama_url=ollama_url,
                timeout_seconds=timeout_seconds,
            )

            if excerpt is None or not excerpt.strip():
                last_failure = HighlightResult(
                    original_pdf_path=pdf_path,
                    highlighted_pdf_path=None,
                    page=candidate_page,
                    success=False,
                    message="No relevant excerpt was selected for highlighting.",
                )
                continue

            match = fuzzy_match_excerpt_to_words(page_words, excerpt)

            if match is None:
                last_failure = HighlightResult(
                    original_pdf_path=pdf_path,
                    highlighted_pdf_path=None,
                    page=candidate_page,
                    success=False,
                    matched_excerpt=excerpt,
                    message="Selected excerpt could not be located on the PDF page.",
                )
                continue

            rects = _line_rects_for_match(page_words, match)

            if not rects:
                last_failure = HighlightResult(
                    original_pdf_path=pdf_path,
                    highlighted_pdf_path=None,
                    page=candidate_page,
                    success=False,
                    matched_excerpt=excerpt,
                    method=match.method,
                    message="Matched excerpt had no highlightable rectangles.",
                )
                continue

            _highlight_rects(pdf_page, rects)
            output_path = _highlight_output_path(pdf_path, output_dir=output_dir)
            document.save(output_path, garbage=4, deflate=True)

            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=output_path,
                page=candidate_page,
                success=True,
                matched_excerpt=excerpt,
                method=match.method,
            )

        if last_failure is not None:
            return last_failure

        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=page,
            success=False,
            message="No relevant excerpt was selected for highlighting.",
        )
    except Exception as exc:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=page,
            success=False,
            message=f"Could not create highlighted PDF: {type(exc).__name__}: {exc}",
        )
    finally:
        document.close()

MetadataHighlightSpec = tuple[str, tuple[float, float, float]]


def _metadata_highlight_output_path(
    pdf_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """Return a temporary output path for a metadata-highlighted PDF copy."""

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "kurrent-metadata-highlights"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{pdf_path.stem}-metadata-{uuid4().hex[:10]}.pdf"


def _highlight_rects_with_color(page, rects, color: tuple[float, float, float]) -> None:
    """Add highlight annotations with a specific RGB color."""

    if not rects:
        return

    try:
        annot = page.add_highlight_annot(rects)

        if annot is not None:
            annot.set_colors(stroke=color)
            annot.update()
            return
    except Exception:
        pass

    for rect in rects:
        annot = page.add_highlight_annot(rect)

        if annot is not None:
            annot.set_colors(stroke=color)
            annot.update()


def _first_metadata_match_on_page(
    page_words: list[PageWord],
    value: str,
) -> WordSpanMatch | None:
    """Return the first exact/fuzzy word-span match for a metadata value."""

    value = collapse_whitespace(value)

    if not value:
        return None

    # Metadata fields are short enough that direct fuzzy matching is appropriate.
    return fuzzy_match_excerpt_to_words(page_words, value, min_score=0.82)


def create_metadata_highlighted_pdf(
    pdf_path: str | Path,
    metadata,
    *,
    output_dir: Path | None = None,
) -> HighlightResult:
    """Create a temporary PDF copy with proposed metadata highlighted.

    Title, authors, year, and DOI are highlighted with distinct colors on the
    first few pages so a user can visually check where Kurrent found each value.
    The original PDF is never modified.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=None,
            success=False,
            message=f"PDF path does not exist: {pdf_path}",
        )

    try:
        import fitz  # noqa: F401
    except ImportError as exc:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=None,
            success=False,
            message=f"PyMuPDF is not available: {exc}",
        )

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=None,
            success=False,
            message=f"Could not open PDF for metadata highlighting: {type(exc).__name__}: {exc}",
        )

    specs: list[MetadataHighlightSpec] = []

    if getattr(metadata, "title", None):
        specs.append((str(metadata.title), (1.0, 0.0, 0.0)))  # red
    if getattr(metadata, "authors", None):
        specs.append((str(metadata.authors), (1.0, 0.55, 0.0)))  # orange
    if getattr(metadata, "year", None) is not None:
        specs.append((str(metadata.year), (0.0, 0.25, 1.0)))  # blue
    if getattr(metadata, "doi", None):
        specs.append((str(metadata.doi), (1.0, 0.35, 0.8)))  # pink

    if not specs:
        document.close()
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=None,
            success=False,
            message="No metadata values were available to highlight.",
        )

    try:
        highlighted_any = False
        max_pages = min(3, document.page_count)

        for value, color in specs:
            for page_index in range(max_pages):
                page = document[page_index]
                page_words = _page_words(page)
                match = _first_metadata_match_on_page(page_words, value)

                if match is None:
                    continue

                rects = _line_rects_for_match(page_words, match)

                if not rects:
                    continue

                _highlight_rects_with_color(page, rects, color)
                highlighted_any = True
                break

        if not highlighted_any:
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=1 if document.page_count else None,
                success=False,
                message="No proposed metadata values could be located on the PDF pages.",
            )

        output_path = _metadata_highlight_output_path(pdf_path, output_dir=output_dir)
        document.save(output_path, garbage=4, deflate=True)

        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=output_path,
            page=1,
            success=True,
            method="metadata-highlight",
        )
    except Exception as exc:
        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=None,
            page=None,
            success=False,
            message=f"Could not create metadata-highlighted PDF: {type(exc).__name__}: {exc}",
        )
    finally:
        document.close()
