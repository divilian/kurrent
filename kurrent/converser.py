"""Stateful Level-1 RAG conversation support for kurrent.

This module deliberately keeps retrieval simple: each turn is sent to the
existing semantic index with only light conversational context, and Ollama is
asked to synthesize a corpus-grounded research-question assessment
from the retrieved chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Callable, Iterable, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from kurrent.cli_display import (
    collapse_whitespace,
    pages_label,
    section_label,
    source_name_for_hit,
)
from kurrent.relevance_judge import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL

__all__ = [
    "DEFAULT_CONVERSE_TOP_K",
    "DEFAULT_CONVERSE_MAX_CONTEXT_CHARS",
    "ConverseError",
    "EvidencePacket",
    "ConverseTurn",
    "ConversationState",
    "ConverseEngine",
    "build_retrieval_query",
    "build_evidence_packets",
    "user_facing_pdf_name",
    "source_label_for_hit",
    "citation_for_hit",
    "format_evidence_packets",
    "build_research_inquiry_messages",
    "call_ollama_chat",
]

DEFAULT_CONVERSE_TOP_K = 8
DEFAULT_CONVERSE_MAX_CONTEXT_CHARS = 14_000
DEFAULT_CHARS_PER_EVIDENCE_PACKET = 1_500
DEFAULT_HISTORY_TURNS = 6


class ConverseError(RuntimeError):
    """Raised when the RAG conversation engine cannot complete a turn."""


class SearcherLike(Protocol):
    """Small protocol for the semantic search dependency used by ConverseEngine."""

    def semantic_chunk_search(
        self,
        search_text: str,
        n_results: int = 10,
        max_distance: float | None = None,
        include_reference_sections: bool = False,
    ) -> list:
        """Return semantic chunk hits for search_text."""


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """One retrieved chunk plus internal provenance for a RAG answer.

    citation is the only source label intended for Ollama-facing prompts. The
    other fields remain available to kurrent for debugging, tests, and future
    /show evidence style commands.
    """

    evidence_id: int
    chunk_id: str
    citation: str
    title: str
    source_name: str | None
    pages: str | None
    section: str | None
    distance: float | None
    text: str


@dataclass(frozen=True, slots=True)
class ConverseTurn:
    """One completed user/assistant turn in a converse session."""

    user_text: str
    retrieval_query: str
    assistant_text: str
    evidence: tuple[EvidencePacket, ...]


@dataclass(slots=True)
class ConversationState:
    """Minimal state carried across a kurrent converse session.

    Level 1 intentionally does not ask the model to rewrite the user's query or
    generate synonym expansions. The summary exists only so follow-up turns like
    "no, I mean tie dissolution specifically" have enough context.
    """

    turns: list[ConverseTurn] = field(default_factory=list)
    max_history_turns: int = DEFAULT_HISTORY_TURNS

    def add_turn(self, turn: ConverseTurn) -> None:
        """Append a turn, retaining only a compact recent history."""

        self.turns.append(turn)

        if len(self.turns) > self.max_history_turns:
            del self.turns[0 : len(self.turns) - self.max_history_turns]

    def compact_summary(self) -> str:
        """Return a compact, deterministic summary of recent session focus."""

        if not self.turns:
            return ""

        lines = []
        for i, turn in enumerate(self.turns, start=1):
            lines.append(f"Turn {i} user focus: {collapse_whitespace(turn.user_text)}")

        return "\n".join(lines)


def build_retrieval_query(user_text: str, state: ConversationState | None = None) -> str:
    """Build the semantic retrieval text for one turn.

    This is deliberately not query expansion. For the first turn, the retrieval
    query is exactly the user's text after whitespace normalization. For later
    turns, recent conversational focus is included so short refinements remain
    meaningful to the embedding model.
    """

    user_text = collapse_whitespace(user_text)

    if state is None:
        return user_text

    summary = state.compact_summary().strip()

    if not summary:
        return user_text

    return (
        f"Recent conversation focus:\n{summary}\n\n"
        f"Current user question or refinement:\n{user_text}"
    )


def _trim_text(text: str, max_chars: int) -> str:
    """Normalize and trim text for prompt inclusion."""

    text = collapse_whitespace(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " [...]"


def _title_for_hit(hit) -> str:
    title = getattr(hit, "title", None)

    if title:
        return str(title)

    source_name = source_name_for_hit(hit)

    if source_name:
        return source_name

    return "unknown document"


MANAGED_PDF_HASH_SUFFIX_RE = re.compile(r"--[0-9a-fA-F]{8,}$")


def user_facing_pdf_name(source_name: str | None) -> str | None:
    """Return a user-facing PDF filename without managed-store hash suffixes."""

    if source_name is None:
        return None

    source_name = str(source_name).strip()

    if not source_name:
        return None

    if not source_name.lower().endswith(".pdf"):
        return source_name

    stem = source_name[:-4]
    stem = MANAGED_PDF_HASH_SUFFIX_RE.sub("", stem)
    return f"{stem}.pdf"


def _normalized_metadata_text(value: object) -> str | None:
    if value is None:
        return None

    text = collapse_whitespace(str(value))
    return text or None


def _metadata_value(
    hit,
    document,
    field_name: str,
) -> str | None:
    """Return metadata from the hit first, then from its parent document."""

    value = getattr(hit, field_name, None)

    if value is None and document is not None:
        value = getattr(document, field_name, None)

    return _normalized_metadata_text(value)


def source_label_for_hit(hit, document=None) -> str:
    """Return the preferred human-facing source label for a semantic hit.

    Prefer author/year metadata. The real Searcher can supply the parent
    document to expose SQLite metadata even though ChunkHit itself may not carry
    authors/year. If author/year are unavailable, fall back first to a cleaned
    PDF filename and then to the title.
    """

    authors = _metadata_value(hit, document, "authors")
    year = _metadata_value(hit, document, "year")

    if authors is not None and year is not None:
        return f"{authors} {year}"

    source_name = user_facing_pdf_name(source_name_for_hit(hit))

    if source_name is not None:
        return source_name

    title = _metadata_value(hit, document, "title")

    if title is not None:
        return title

    return "unknown document"


def citation_for_hit(hit, document=None) -> str:
    """Return the exact citation phrase Ollama should use for this hit."""

    citation = source_label_for_hit(hit, document=document)
    pages = pages_label(hit)

    if pages is not None:
        citation = f"{citation}, {pages}"

    return citation


DocumentLookup = Callable[[object], object | None]


def build_evidence_packets(
    hits: Iterable,
    chars_per_packet: int = DEFAULT_CHARS_PER_EVIDENCE_PACKET,
    document_lookup: DocumentLookup | None = None,
) -> tuple[EvidencePacket, ...]:
    """Convert semantic hits into prompt-ready evidence packets.

    document_lookup lets callers provide parent-document metadata from SQLite
    without forcing ChunkHit itself to duplicate every document field.
    """

    packets: list[EvidencePacket] = []

    for i, hit in enumerate(hits, start=1):
        document = None

        if document_lookup is not None:
            document = document_lookup(hit)

        packets.append(
            EvidencePacket(
                evidence_id=i,
                chunk_id=hit.chunk_id,
                citation=citation_for_hit(hit, document=document),
                title=_title_for_hit(hit),
                source_name=source_name_for_hit(hit),
                pages=pages_label(hit),
                section=section_label(hit),
                distance=getattr(hit, "distance", None),
                text=_trim_text(hit.text, chars_per_packet),
            )
        )

    return tuple(packets)


def format_evidence_packets(
    evidence: Iterable[EvidencePacket],
    max_context_chars: int = DEFAULT_CONVERSE_MAX_CONTEXT_CHARS,
) -> str:
    """Format retrieved evidence as minimal JSON for the Ollama prompt.

    The model sees only the citation string it should use and the excerpt it
    should reason from. Internal kurrent details such as evidence IDs, chunk IDs,
    semantic distances, raw managed filenames, and section labels are omitted
    from the Ollama-facing prompt.
    """

    items: list[dict[str, str]] = []

    for packet in evidence:
        item = {
            "citation": packet.citation,
            "excerpt": packet.text,
        }
        candidate_items = [*items, item]
        rendered = json.dumps(
            candidate_items,
            indent=2,
            ensure_ascii=False,
        )

        if items and len(rendered) > max_context_chars:
            break

        items.append(item)

    return json.dumps(
        items,
        indent=2,
        ensure_ascii=False,
    )


def build_research_inquiry_messages(
    user_text: str,
    evidence: Iterable[EvidencePacket],
    state: ConversationState | None = None,
    max_context_chars: int = DEFAULT_CONVERSE_MAX_CONTEXT_CHARS,
) -> list[dict[str, str]]:
    """Build Ollama chat messages for corpus-grounded RQ discovery."""

    summary = "" if state is None else state.compact_summary().strip()
    evidence_text = format_evidence_packets(
        evidence,
        max_context_chars=max_context_chars,
    )

    system_message = """
You are Kurrent's Research Inquiry assistant. Help the user assess how their
research question relates to the currently ingested Kurrent corpus.

Rules:
- Use only the provided JSON evidence array as literature evidence.
- Treat each evidence object's citation field as the exact source phrase to use.
- Do not write "Evidence 1", "Evidence 2", chunk IDs, distances, or filenames
  other than the citation strings provided.
- Make corpus-scoped claims, not global claims about all scholarship.
- Distinguish whether the evidence directly addresses, partially addresses, is
  merely adjacent to, or does not address the proposed research question.
- Do not invent papers, authors, page numbers, or claims not present in evidence.
- If evidence is weak or absent, say so plainly.
""".strip()

    user_message = f"""
Recent session summary:
{summary if summary else "(none)"}

Current user research interest or refinement:
{collapse_whitespace(user_text)}

Retrieved Kurrent evidence as JSON:
{evidence_text}

Task:
Write a concise research-literature assessment using this structure:

1. Closest Kurrent evidence
   List the closest citation(s) and explain what each does and does not show
   about the proposed research question.

2. Possible novelty angle
   Suggest a cautious corpus-scoped novelty claim if the evidence supports one.

Do not include a separate search-terms, further-reading, or papers-to-ingest
section unless the user explicitly asks for one.
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def call_ollama_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 120.0,
) -> str:
    """Call Ollama's chat API and return the assistant message content."""

    api_url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
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
        raise ConverseError(f"Ollama chat unavailable: {type(exc).__name__}: {exc}") from exc

    content = response_data.get("message", {}).get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise ConverseError("Ollama returned an empty converse response.")

    return content.strip()


AnswerFunction = Callable[[list[dict[str, str]]], str]
ProgressCallback = Callable[[str], None]


class ConverseEngine:
    """Level-1 stateful RAG conversation engine."""

    def __init__(
        self,
        searcher: SearcherLike,
        model: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 120.0,
        top_k: int = DEFAULT_CONVERSE_TOP_K,
        max_distance: float | None = None,
        include_reference_sections: bool = False,
        max_context_chars: int = DEFAULT_CONVERSE_MAX_CONTEXT_CHARS,
        answer_function: AnswerFunction | None = None,
    ) -> None:
        self.searcher = searcher
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k
        self.max_distance = max_distance
        self.include_reference_sections = include_reference_sections
        self.max_context_chars = max_context_chars
        self.answer_function = answer_function
        self.state = ConversationState()

    def _document_for_hit(self, hit):
        """Return the parent document for a hit when the searcher exposes state."""

        state_store = getattr(self.searcher, "state_store", None)

        if state_store is None or not hasattr(state_store, "get_document"):
            return None

        try:
            return state_store.get_document(hit.doc_id)
        except Exception:
            return None

    def answer_user_turn(
        self,
        user_text: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ConverseTurn:
        """Retrieve Kurrent evidence and answer one user turn.

        progress_callback receives short user-facing status messages between
        slow stages. It is intentionally optional so tests and non-CLI callers
        can use the engine silently.
        """

        def report(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)

        user_text = collapse_whitespace(user_text)

        if not user_text:
            raise ConverseError("Converse requires a non-empty user turn.")

        report("Preparing retrieval query...")
        retrieval_query = build_retrieval_query(user_text, self.state)

        report("Searching Kurrent semantic index...")
        hits = self.searcher.semantic_chunk_search(
            retrieval_query,
            n_results=self.top_k,
            max_distance=self.max_distance,
            include_reference_sections=self.include_reference_sections,
        )

        hit_count = len(hits)
        chunk_word = "chunk" if hit_count == 1 else "chunks"
        report(f"Retrieved {hit_count} candidate {chunk_word}.")

        report("Building evidence packet for Ollama..." if hit_count == 1 else "Building evidence packets for Ollama...")
        evidence = build_evidence_packets(
            hits,
            document_lookup=self._document_for_hit,
        )
        messages = build_research_inquiry_messages(
            user_text=user_text,
            evidence=evidence,
            state=self.state,
            max_context_chars=self.max_context_chars,
        )

        report("Asking Ollama for a corpus-grounded assessment...")
        if self.answer_function is not None:
            assistant_text = self.answer_function(messages)
        else:
            assistant_text = call_ollama_chat(
                messages,
                model=self.model,
                ollama_url=self.ollama_url,
                timeout_seconds=self.timeout_seconds,
            )

        report("Recording this turn in the conversation state...")
        turn = ConverseTurn(
            user_text=user_text,
            retrieval_query=retrieval_query,
            assistant_text=assistant_text,
            evidence=evidence,
        )
        self.state.add_turn(turn)
        return turn
