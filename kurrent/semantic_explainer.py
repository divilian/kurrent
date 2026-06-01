"""Ollama-based explanations for semantic chunk search results."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import json
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from kurrent.cli_display import (
    collapse_whitespace,
    pages_label,
    section_label,
    source_name_for_hit,
)

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
    "ChunkExplanation",
    "explain_chunk_with_ollama",
    "SemanticExplanationBuffer",
]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = os.environ.get(
    "KURRENT_OLLAMA_MODEL",
    "llama3.1:8b-instruct-q4_K_M",
)


@dataclass(frozen=True, slots=True)
class ChunkExplanation:
    """Ollama explanation of how a chunk relates to a semantic query."""

    relevant: bool | None
    explanation: str
    error: str | None = None


def _ollama_chat_json(
    messages: list[dict[str, str]],
    model: str,
    ollama_url: str,
    timeout_seconds: float,
) -> dict:
    """Call Ollama's chat API and return parsed JSON content."""

    api_url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    content = response_data.get("message", {}).get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty explanation.")

    return json.loads(content)


def _build_chunk_explanation_prompt(query: str, hit) -> list[dict[str, str]]:
    """Build a compact Ollama prompt for explaining one semantic hit."""

    source = source_name_for_hit(hit) or "unknown source"
    section = section_label(hit) or "unknown section"
    pages = pages_label(hit) or "unknown pages"
    chunk_text = collapse_whitespace(hit.text)

    system_message = (
        "You explain why a semantically retrieved academic text chunk may or "
        "may not relate to a user's search query. Return only JSON. Write "
        "compact notes, not prose introductions."
    )
    user_message = f"""
User query:
{query}

Chunk context:
source: {source}
section: {section}
pages: {pages}

Chunk text:
{chunk_text}

Task:
Explain how this chunk relates to the user query in a compact note, ideally
5-25 words. Do not begin with phrases such as "The chunk discusses", "This
chunk discusses", "The passage discusses", or "Discusses". Prefer direct
wording like "Links homophily to polarization through repeated like-with-like
interaction." If the chunk is not actually relevant, only weakly relevant,
just a table of contents entry, bibliography entry, header/footer, or otherwise
not substantive, set relevant to false and say why. Otherwise set relevant to
true.

Return exactly this JSON shape:
{{
  "relevant": true,
  "explanation": "..."
}}
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]



def _clean_ollama_explanation(text: str) -> str:
    """Remove repetitive Ollama lead-ins from a relevance explanation."""

    text = collapse_whitespace(text)
    text = re.sub(
        r"^(?:the|this)\s+(?:chunk|passage|excerpt|text)\s+"
        r"(?:discusses|describes|explains|shows|argues|mentions|covers|"
        r"focuses on|relates to|is about)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:discusses|describes|explains|shows|argues|mentions|covers|"
        r"focuses on|relates to)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip()

    if not text:
        return text

    return text[0].upper() + text[1:]

def explain_chunk_with_ollama(
    query: str,
    hit,
    model: str,
    ollama_url: str,
    timeout_seconds: float,
) -> ChunkExplanation:
    """Ask Ollama how a semantic chunk hit relates to the query."""

    try:
        data = _ollama_chat_json(
            _build_chunk_explanation_prompt(query, hit),
            model=model,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return ChunkExplanation(
            relevant=None,
            explanation="Ollama explanation unavailable.",
            error=f"{type(exc).__name__}: {exc}",
        )

    relevant = data.get("relevant")
    explanation = data.get("explanation")

    if not isinstance(relevant, bool):
        relevant = None

    if not isinstance(explanation, str) or not explanation.strip():
        return ChunkExplanation(
            relevant=relevant,
            explanation="Ollama returned no usable explanation.",
            error="Missing explanation field.",
        )

    return ChunkExplanation(
        relevant=relevant,
        explanation=_clean_ollama_explanation(explanation),
    )


class SemanticExplanationBuffer:
    """Background producer for Ollama chunk explanations."""

    def __init__(
        self,
        query: str,
        hits,
        model: str,
        ollama_url: str,
        timeout_seconds: float = 45.0,
        max_workers: int = 2,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: dict[str, Future] = {}

        for hit in hits:
            self.futures[hit.chunk_id] = self.executor.submit(
                explain_chunk_with_ollama,
                query,
                hit,
                model,
                ollama_url,
                timeout_seconds,
            )

    def get(self, hit, wait_seconds: float = 0.0) -> ChunkExplanation | None:
        """Return this hit's explanation, waiting briefly if requested."""

        future = self.futures.get(hit.chunk_id)

        if future is None:
            return None

        try:
            return future.result(timeout=wait_seconds)
        except FutureTimeoutError:
            return None

    def close(self) -> None:
        """Cancel pending work and let the CLI exit promptly."""

        self.executor.shutdown(wait=False, cancel_futures=True)


