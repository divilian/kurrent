"""Version fingerprints for kurrent's derived-text pipeline.

The chunker version alone is not enough to decide whether stored chunks are
current. Chunk text also depends on the PDF extractor, section-heading pipeline,
and sectioning mode. When any of those pieces changes, re-ingesting an existing
PDF should rebuild the derived artifacts even if the nominal chunker version has
not changed.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

__all__ = [
    "CHUNKER_ALGORITHM_VERSION",
    "LLM_SECTIONER_VERSION",
    "PDF_TEXT_EXTRACTOR_VERSION",
    "PIPELINE_FINGERPRINT_VERSION",
    "SECTIONER_VERSION",
    "current_text_pipeline_fingerprint",
]


PDF_TEXT_EXTRACTOR_VERSION = "layout-aware-pymupdf-v2"
SECTIONER_VERSION = "sectioner-v4"
LLM_SECTIONER_VERSION = "ollama-section-headings-v2"
CHUNKER_ALGORITHM_VERSION = "section-aware-fixed-char-v2"
PIPELINE_FINGERPRINT_VERSION = "text-pipeline-fingerprint-v1"


def _normalized_reviewed_headings_fingerprint(
    reviewed_headings: Sequence[str] | None,
) -> str:
    """Return a stable fingerprint for a caller-supplied heading list."""

    if reviewed_headings is None:
        return "none"

    normalized = [" ".join(heading.split()) for heading in reviewed_headings]
    payload = "\n".join(normalized)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sectioning_mode(
    reviewed_headings: Sequence[str] | None,
    use_llm_sectioning: bool,
) -> str:
    """Return the effective sectioning mode for this ingest run."""

    if reviewed_headings is not None:
        return "reviewed-headings"

    if use_llm_sectioning:
        return "llm-assisted"

    return "rules-based"


def current_text_pipeline_fingerprint(
    reviewed_headings: Sequence[str] | None = None,
    use_llm_sectioning: bool = True,
    target_chars: int = 2000,
    extractor_version: str = PDF_TEXT_EXTRACTOR_VERSION,
    sectioner_version: str = SECTIONER_VERSION,
    llm_sectioner_version: str = LLM_SECTIONER_VERSION,
    chunker_algorithm_version: str = CHUNKER_ALGORITHM_VERSION,
) -> str:
    """Return the fingerprint for the current PDF-to-chunks text pipeline.

    This intentionally includes more than the chunker. Re-ingesting a PDF should
    rebuild chunks if extraction, section recognition, sectioning mode, reviewed
    headings, or chunk sizing has changed.
    """

    mode = _sectioning_mode(
        reviewed_headings=reviewed_headings,
        use_llm_sectioning=use_llm_sectioning,
    )
    reviewed_heading_hash = _normalized_reviewed_headings_fingerprint(
        reviewed_headings,
    )

    parts = [
        f"fingerprint={PIPELINE_FINGERPRINT_VERSION}",
        f"extractor={extractor_version}",
        f"sectioner={sectioner_version}",
        f"llm_sectioner={llm_sectioner_version}",
        f"sectioning_mode={mode}",
        f"reviewed_headings={reviewed_heading_hash}",
        f"chunker={chunker_algorithm_version}",
        f"target_chars={target_chars}",
    ]

    return ";".join(parts)
