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
from kurrent.relevance_judge import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL

DEFAULT_HIGHLIGHT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_PAGE_TEXT_CHARS = 12_000
DEFAULT_MIN_MATCH_SCORE = 0.72
DEFAULT_MIN_EXCERPT_WORDS = 12
DEFAULT_MAX_EXCERPT_WORDS = 120


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


def select_relevant_excerpt_with_ollama(
    page_text: str,
    research_interest: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = DEFAULT_HIGHLIGHT_TIMEOUT_SECONDS,
) -> str | None:
    """Ask Ollama for an exact relevant excerpt copied from page_text."""

    page_text = collapse_whitespace(page_text)
    research_interest = collapse_whitespace(research_interest)

    if not page_text or not research_interest:
        return None

    if len(page_text) > DEFAULT_MAX_PAGE_TEXT_CHARS:
        page_text = page_text[:DEFAULT_MAX_PAGE_TEXT_CHARS].rstrip() + " [...]"

    system_message = (
        "You select exact text spans from academic PDF pages. Return only JSON. "
        "Do not paraphrase."
    )
    user_message = f"""
Research interest:
{research_interest}

Page text:
{page_text}

Task:
Select the shortest contiguous excerpt from the page text that is most relevant
to the research interest. The excerpt must be copied verbatim from the page text,
except that whitespace may be collapsed. Prefer 25-120 words. If no passage is
meaningfully relevant, return an empty excerpt.

Return exactly this JSON shape:
{{"excerpt": "..."}}
""".strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
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
        page_index = page - 1

        if page_index < 0 or page_index >= document.page_count:
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=page,
                success=False,
                message=f"Page {page} is outside the PDF page range.",
            )

        pdf_page = document[page_index]
        page_words = _page_words(pdf_page)

        if not page_words:
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=page,
                success=False,
                message="No extractable words were found on this PDF page.",
            )

        page_text = clean_page_text_from_words(page_words)

        if excerpt_selector is None:
            excerpt_selector = lambda clean_text, interest: select_relevant_excerpt_with_ollama(
                clean_text,
                interest,
                model=model,
                ollama_url=ollama_url,
                timeout_seconds=timeout_seconds,
            )

        excerpt = excerpt_selector(page_text, research_interest)

        if excerpt is None and fallback_excerpt is not None:
            excerpt = _truncate_word_count(
                fallback_excerpt,
                min_words=DEFAULT_MIN_EXCERPT_WORDS,
                max_words=DEFAULT_MAX_EXCERPT_WORDS,
            )

        if excerpt is None or not excerpt.strip():
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=page,
                success=False,
                message="No relevant excerpt was selected for highlighting.",
            )

        match = fuzzy_match_excerpt_to_words(page_words, excerpt)

        if match is None:
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=page,
                success=False,
                matched_excerpt=excerpt,
                message="Selected excerpt could not be located on the PDF page.",
            )

        rects = _line_rects_for_match(page_words, match)

        if not rects:
            return HighlightResult(
                original_pdf_path=pdf_path,
                highlighted_pdf_path=None,
                page=page,
                success=False,
                matched_excerpt=excerpt,
                method=match.method,
                message="Matched excerpt had no highlightable rectangles.",
            )

        _highlight_rects(pdf_page, rects)
        output_path = _highlight_output_path(pdf_path, output_dir=output_dir)
        document.save(output_path, garbage=4, deflate=True)

        return HighlightResult(
            original_pdf_path=pdf_path,
            highlighted_pdf_path=output_path,
            page=page,
            success=True,
            matched_excerpt=excerpt,
            method=match.method,
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
