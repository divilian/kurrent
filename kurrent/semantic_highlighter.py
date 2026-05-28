"""Semantic excerpt selection and highlighting helpers."""

from __future__ import annotations

import math
import re

from kurrent.cli_display import (
    ANSI_BOLD,
    ANSI_BOLD_RED,
    ANSI_BOLD_YELLOW,
    ANSI_RESET,
    ansi_enabled,
    collapse_whitespace,
    context_window,
)


SEMANTIC_HIGHLIGHT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "might", "more",
    "most", "no", "not", "of", "on", "or", "our", "out", "over", "she",
    "should", "so", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "under",
    "up", "was", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your",
}

SEMANTIC_HIGHLIGHT_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'-]{2,}\b")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return cosine similarity for two embedding vectors."""

    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def semantic_windows(text: str, words_per_window: int = 70) -> list[str]:
    """Return overlapping word windows for choosing a semantic excerpt."""

    words = collapse_whitespace(text).split()

    if not words:
        return []

    if len(words) <= words_per_window:
        return [" ".join(words)]

    stride = max(1, words_per_window // 2)
    windows = []

    for start in range(0, len(words), stride):
        window_words = words[start:start + words_per_window]

        if len(window_words) < 12 and windows:
            break

        windows.append(" ".join(window_words))

        if start + words_per_window >= len(words):
            break

    return windows


def best_semantic_excerpt(
    text: str,
    query: str,
    embedder,
    max_chars: int,
) -> str:
    """Return the chunk excerpt whose local window best matches the query."""

    collapsed = collapse_whitespace(text)

    if len(collapsed) <= max_chars:
        return collapsed

    windows = semantic_windows(collapsed)

    if not windows:
        return context_window(collapsed, None, width=max_chars)

    embeddings = embedder.generate_embeddings([query] + windows)
    query_embedding = embeddings[0]
    window_embeddings = embeddings[1:]

    best_index = max(
        range(len(windows)),
        key=lambda i: cosine_similarity(query_embedding, window_embeddings[i]),
    )
    best_window = windows[best_index]
    best_start = collapsed.find(best_window)

    if best_start < 0:
        return context_window(collapsed, None, width=max_chars)

    best_center = best_start + len(best_window) // 2
    start = max(0, best_center - max_chars // 2)
    end = min(len(collapsed), start + max_chars)
    start = max(0, end - max_chars)

    excerpt = collapsed[start:end].strip()

    if start > 0:
        excerpt = "[...] " + excerpt

    if end < len(collapsed):
        excerpt = excerpt + " [...]"

    return excerpt


def semantic_candidate_words(text: str) -> list[str]:
    """Return unique content words eligible for semantic highlighting."""

    words: list[str] = []
    seen: set[str] = set()

    for match in SEMANTIC_HIGHLIGHT_TOKEN_RE.finditer(text):
        word = match.group(0)
        key = word.lower().strip("'-")

        if len(key) < 4:
            continue

        if key in SEMANTIC_HIGHLIGHT_STOPWORDS:
            continue

        if key in seen:
            continue

        seen.add(key)
        words.append(word)

    return words


def semantic_highlight_tiers(
    text: str,
    query: str,
    embedder,
) -> dict[str, str]:
    """Assign candidate words to bold/yellow/red semantic-highlight tiers."""

    candidates = semantic_candidate_words(text)

    if not candidates:
        return {}

    embeddings = embedder.generate_embeddings([query] + candidates)
    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    scored = []

    for word, embedding in zip(candidates, candidate_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scored.append((word.lower().strip("'-"), score))

    scored.sort(key=lambda item: item[1], reverse=True)

    if not scored or scored[0][1] < 0.12:
        return {}

    highlight_count = min(18, max(5, math.ceil(len(scored) * 0.18)))
    highlighted = scored[:highlight_count]

    red_count = max(1, math.ceil(len(highlighted) * 0.15))
    yellow_count = max(1, math.ceil(len(highlighted) * 0.30))

    tiers: dict[str, str] = {}

    for i, (word, score) in enumerate(highlighted):
        if score < 0.12:
            continue

        if i < red_count:
            tiers[word] = "red"
        elif i < red_count + yellow_count:
            tiers[word] = "yellow"
        else:
            tiers[word] = "bold"

    return tiers


def apply_semantic_highlights(text: str, tiers: dict[str, str]) -> str:
    """Apply semantic-highlight tiers to matching words in display text."""

    if not ansi_enabled() or not tiers:
        return text

    def replace(match: re.Match) -> str:
        word = match.group(0)
        key = word.lower().strip("'-")
        tier = tiers.get(key)

        if tier == "red":
            return f"{ANSI_BOLD_RED}{word}{ANSI_RESET}"

        if tier == "yellow":
            return f"{ANSI_BOLD_YELLOW}{word}{ANSI_RESET}"

        if tier == "bold":
            return f"{ANSI_BOLD}{word}{ANSI_RESET}"

        return word

    return SEMANTIC_HIGHLIGHT_TOKEN_RE.sub(replace, text)


def semantically_highlighted_excerpt(
    text: str,
    query: str,
    embedder,
    max_chars: int,
) -> str:
    """Return a semantic excerpt with three-tier semantic word highlighting."""

    excerpt = best_semantic_excerpt(
        text,
        query,
        embedder,
        max_chars=max_chars,
    )
    tiers = semantic_highlight_tiers(excerpt, query, embedder)
    return apply_semantic_highlights(excerpt, tiers)


def semantically_highlighted_text(text: str, query: str, embedder) -> str:
    """Return full text with semantic word highlighting applied."""

    collapsed = collapse_whitespace(text)
    tiers = semantic_highlight_tiers(collapsed, query, embedder)
    return apply_semantic_highlights(collapsed, tiers)


