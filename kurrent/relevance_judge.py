"""LLM-based relevance judgments for semantic chunk search results."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import json
import re
from kurrent.cli_display import (
    collapse_whitespace,
    pages_label,
    section_label,
    source_name_for_hit,
)
from kurrent.config import DEFAULT_RELEVANCE_LLM
from kurrent.llm_provider import (
    LLMCallError,
    LLMConfigurationError,
    call_llm_chat,
)

__all__ = [
    "DEFAULT_RELEVANCE_LLM",
    "RelevanceJudgment",
    "judge_chunk_relevance",
    "RelevanceJudgmentBuffer",
]

@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """LLM judgment of whether a candidate chunk satisfies a semantic query."""

    relevant: bool | None
    explanation: str
    relevance: str | None = None
    answers_query: bool | None = None
    missing_concepts: tuple[str, ...] = ()
    error: str | None = None


def _llm_chat_json(
    messages: list[dict[str, str]],
    model: str,
    timeout_seconds: float,
) -> dict:
    """Call the configured chat provider and return parsed JSON content."""

    content = call_llm_chat(
        messages,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=0,
        json_format=True,
    )

    if not content.strip():
        raise ValueError("LLM returned an empty relevance judgment.")

    return json.loads(content)

def _build_relevance_judgment_prompt(query: str, hit) -> list[dict[str, str]]:
    """Build a compact LLM prompt for judging one semantic hit."""

    source = source_name_for_hit(hit) or "unknown source"
    section = section_label(hit) or "unknown section"
    pages = pages_label(hit) or "unknown pages"
    chunk_text = collapse_whitespace(hit.text)

    system_message = (
        "You are a relevance judge for an academic literature-search tool. "
        "A vector search system has retrieved a candidate text chunk. Your job "
        "is to decide whether the chunk would be a useful literature-search hit "
        "for the user's query. Return only JSON."
    )
    user_message = f"""
User query:
{query}

Candidate chunk context:
source: {source}
section: {section}
pages: {pages}

Candidate chunk text:
{chunk_text}

Task:
Judge whether this candidate chunk is a useful academic literature-search hit
for the user's query. The query may be a question, a claim, a topic phrase, a
method, a mechanism, or a named concept. Do not require the chunk to answer a
complete question if the user's query is only a search phrase.

Use these relevance labels:
- "strong": the chunk directly discusses, uses, instantiates, defines, measures,
  analyzes, or gives important evidence about the specific concept, method,
  process, relationship, or claim in the query.
- "partial": the chunk mentions some important query concepts, but misses a
  central concept, relation, direction, causal mechanism, or claimed process.
- "weak": the chunk is only topically adjacent or shares isolated vocabulary.
- "none": the chunk is not substantively relevant, or is only a bibliography,
  header/footer, table of contents entry, metadata, or other non-body artifact.

Use literature-search judgment, not QA judgment. For example, if the query is
"an SBM-generated network", a chunk that says an assortative stochastic block
model was used to create an initial network is a strong match, even though the
query is not phrased as a question. If the query asks about dissolving network
ties due to homophily, a chunk that mentions homophily but not tie dissolution,
edge deletion, rewiring, or a similar process should be partial or weak, not
strong.

Set:
- relevant=true only for "strong" matches.
- relevant=false for "partial", "weak", and "none".
- answers_query=true when the chunk substantially satisfies the search intent,
  either by answering a question or by directly matching a search phrase.
- missing_concepts to a short list of important query concepts or relationships
  absent from the chunk.
- explanation to a compact note, ideally 5-25 words. Do not begin with phrases
  such as "The chunk discusses", "This chunk discusses", "The passage discusses",
  or "Discusses". Prefer direct wording.

Return exactly this JSON shape:
{{
  "relevance": "strong",
  "relevant": true,
  "answers_query": true,
  "missing_concepts": [],
  "explanation": "..."
}}
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _clean_judgment_explanation(text: str) -> str:
    """Remove repetitive LLM lead-ins from a relevance explanation."""

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


def _normalize_relevance_label(value: object) -> str | None:
    """Normalize a model-provided relevance label."""

    if not isinstance(value, str):
        return None

    value = value.strip().lower()

    if value in {"strong", "partial", "weak", "none"}:
        return value

    return None


def _normalize_missing_concepts(value: object) -> tuple[str, ...]:
    """Normalize a model-provided missing concept list."""

    if not isinstance(value, list):
        return ()

    concepts = []

    for item in value:
        if isinstance(item, str) and item.strip():
            concepts.append(collapse_whitespace(item))

    return tuple(concepts)


def judge_chunk_relevance(
    query: str,
    hit,
    model: str,
    timeout_seconds: float,
) -> RelevanceJudgment:
    """Ask the configured LLM whether a semantic chunk hit truly satisfies the query."""

    try:
        data = _llm_chat_json(
            _build_relevance_judgment_prompt(query, hit),
            model=model,
            timeout_seconds=timeout_seconds,
        )
    except (
        OSError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
        LLMConfigurationError,
        LLMCallError,
    ) as exc:
        return RelevanceJudgment(
            relevant=None,
            relevance=None,
            answers_query=None,
            explanation="LLM relevance judgment unavailable.",
            error=f"{type(exc).__name__}: {exc}",
        )

    relevance = _normalize_relevance_label(data.get("relevance"))
    relevant = data.get("relevant")
    answers_query = data.get("answers_query")
    explanation = data.get("explanation")
    missing_concepts = _normalize_missing_concepts(data.get("missing_concepts"))

    if not isinstance(relevant, bool):
        relevant = None

    if not isinstance(answers_query, bool):
        answers_query = None

    if relevance is not None:
        # The label is the authoritative strict gate. Only strong matches pass.
        relevant = relevance == "strong"

        if answers_query is None:
            answers_query = relevance == "strong"

    if not isinstance(explanation, str) or not explanation.strip():
        return RelevanceJudgment(
            relevant=relevant,
            relevance=relevance,
            answers_query=answers_query,
            missing_concepts=missing_concepts,
            explanation="LLM returned no usable relevance explanation.",
            error="Missing explanation field.",
        )

    return RelevanceJudgment(
        relevant=relevant,
        relevance=relevance,
        answers_query=answers_query,
        missing_concepts=missing_concepts,
        explanation=_clean_judgment_explanation(explanation),
    )


class RelevanceJudgmentBuffer:
    """Background producer for LLM relevance judgments."""

    def __init__(
        self,
        query: str,
        hits,
        model: str,
        timeout_seconds: float = 45.0,
        max_workers: int = 2,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: dict[str, Future] = {}

        for hit in hits:
            self.futures[hit.chunk_id] = self.executor.submit(
                judge_chunk_relevance,
                query,
                hit,
                model,
                timeout_seconds,
            )

    def get(self, hit, wait_seconds: float = 0.0) -> RelevanceJudgment | None:
        """Return this hit's relevance judgment, waiting briefly if requested."""

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
