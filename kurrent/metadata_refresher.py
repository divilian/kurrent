"""Batch metadata refresh helpers for kurrent documents.

This module is intentionally conservative. It identifies document metadata that
looks obviously bad or incomplete, then proposes better metadata using this
order of evidence:

1. DOI found in existing metadata or early PDF text, resolved through Crossref.
2. Ollama extraction from early PDF text when Crossref is unavailable/incomplete.

The public functions return proposed updates separately from applying them so
CLI callers can support dry-run, interactive review, and batch apply modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from kurrent.cli_display import collapse_whitespace
from kurrent.metadata_extractor import (
    clean_author_metadata_text,
    clean_metadata_text,
    clean_title_metadata_text,
    extract_doi,
    extract_text_from_first_pages,
    lookup_crossref_metadata,
    looks_like_bad_title,
)
from kurrent.config import DEFAULT_METADATA_LLM, DEFAULT_OLLAMA_URL
from kurrent.schema import Document, ExtractedMetadata

__all__ = [
    "MetadataRefreshError",
    "MetadataAssessment",
    "MetadataRefreshProposal",
    "MetadataRefreshResult",
    "assess_document_metadata",
    "metadata_updates_for_document",
    "extract_metadata_with_ollama",
    "propose_metadata_refresh",
    "apply_metadata_refresh",
    "ensure_ollama_available",
    "refresh_documents_metadata",
]

RefreshMethod = Literal["auto", "crossref", "llm"]
MetadataSource = Literal["crossref", "llm", "none"]
ProgressCallback = Callable[[str], None]

DEFAULT_OLLAMA_MODEL = DEFAULT_METADATA_LLM

CURRENT_YEAR_CEILING = 2100
BAD_AUTHOR_VALUES = {
    "author",
    "authors",
    "unknown",
    "unknown author",
    "anonymous",
    "admin",
    "administrator",
    "user",
    "windows user",
    "microsoft word",
    "pdf",
}

ZOTERO_STORAGE_DIR_RE = re.compile(r"^[A-Z0-9]{8}$")
SHORT_GIBBERISH_RE = re.compile(r"^[A-Za-z]{2,}\d{2,}$|^[A-Z0-9_-]{4,16}$")
DOI_PREFIX_RE = re.compile(r"^10\.\d{4,9}/", re.IGNORECASE)


class MetadataRefreshError(RuntimeError):
    """Raised when metadata refresh cannot complete."""


@dataclass(frozen=True, slots=True)
class MetadataAssessment:
    """Assessment of existing document metadata quality."""

    needs_refresh: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetadataRefreshProposal:
    """Proposed metadata replacement for a document."""

    metadata: ExtractedMetadata
    source: MetadataSource
    confidence: str
    reason: str


@dataclass(frozen=True, slots=True)
class MetadataRefreshResult:
    """Result of inspecting one document for metadata refresh."""

    document: Document
    assessment: MetadataAssessment
    proposal: MetadataRefreshProposal | None
    updates: dict[str, object]
    updated_document: Document | None = None
    error: str | None = None



def _norm(value: object) -> str | None:
    if value is None:
        return None
    return clean_metadata_text(str(value))



def _is_missing(value: object) -> bool:
    return _norm(value) is None



def _looks_like_filename_or_storage_junk(value: str | None) -> bool:
    value = _norm(value)

    if value is None:
        return True

    lower_value = value.lower()

    if lower_value.endswith(".pdf"):
        return True

    if "/" in value or "\\" in value:
        return True

    stem = Path(value).stem

    if ZOTERO_STORAGE_DIR_RE.fullmatch(stem):
        return True

    if SHORT_GIBBERISH_RE.fullmatch(stem) and len(stem) <= 12:
        return True

    return False



def _looks_like_bad_authors(authors: str | None) -> bool:
    authors = _norm(authors)

    if authors is None:
        return True

    lower_authors = authors.lower()

    if lower_authors in BAD_AUTHOR_VALUES:
        return True

    if lower_authors.endswith(".pdf"):
        return True

    if _looks_like_filename_or_storage_junk(authors):
        return True

    if DOI_PREFIX_RE.match(authors):
        return True

    # Very long author strings are often pasted abstracts, affiliations, or bad
    # PDF metadata rather than author lists.
    if len(authors) > 300:
        return True

    return False



def _looks_like_bad_year(year: int | None) -> bool:
    if year is None:
        return True

    try:
        year = int(year)
    except (TypeError, ValueError):
        return True

    return not 1800 <= year <= CURRENT_YEAR_CEILING



def assess_document_metadata(document: Document) -> MetadataAssessment:
    """Return whether document metadata looks obviously bad or incomplete."""

    reasons: list[str] = []

    title = _norm(document.title)
    authors = _norm(document.authors)

    if looks_like_bad_title(title):
        reasons.append("title is missing or looks like PDF/application metadata")
    elif _looks_like_filename_or_storage_junk(title):
        reasons.append("title looks like a filename or storage identifier")

    if _looks_like_bad_authors(authors):
        reasons.append("authors are missing or look bogus")

    if _looks_like_bad_year(document.year):
        reasons.append("year is missing or implausible")

    return MetadataAssessment(
        needs_refresh=bool(reasons),
        reasons=tuple(reasons),
    )



def _field_is_bad(document: Document, field_name: str) -> bool:
    if field_name == "title":
        title = _norm(document.title)
        return looks_like_bad_title(title) or _looks_like_filename_or_storage_junk(title)

    if field_name == "authors":
        return _looks_like_bad_authors(_norm(document.authors))

    if field_name == "year":
        return _looks_like_bad_year(document.year)

    if field_name == "doi":
        return _is_missing(document.doi)

    raise ValueError(f"Unknown metadata field: {field_name}")



def _proposal_field(metadata: ExtractedMetadata, field_name: str) -> object:
    return getattr(metadata, field_name)



def metadata_updates_for_document(
    document: Document,
    proposal: MetadataRefreshProposal,
    replace_all: bool = False,
) -> dict[str, object]:
    """Return safe metadata updates from proposal for document.

    By default, only fields that look bad/missing in the current document are
    replaced. Crossref proposals may be applied with replace_all=True by callers
    that want a full high-confidence metadata refresh.
    """

    updates: dict[str, object] = {}

    for field_name in ["title", "authors", "year", "doi"]:
        proposed_value = _proposal_field(proposal.metadata, field_name)

        if field_name == "title":
            proposed_value = clean_title_metadata_text(proposed_value)
        elif field_name == "authors":
            proposed_value = clean_author_metadata_text(proposed_value)

        if proposed_value is None:
            continue

        if not replace_all and not _field_is_bad(document, field_name):
            continue

        current_value = getattr(document, field_name)

        if current_value == proposed_value:
            continue

        updates[field_name] = proposed_value

    return updates



def _metadata_has_core_fields(metadata: ExtractedMetadata) -> bool:
    return metadata.title is not None and metadata.authors is not None and metadata.year is not None



def _best_doi_for_document(document: Document, early_text: str) -> str | None:
    doi = _norm(document.doi)

    if doi is not None:
        return doi

    return extract_doi(early_text)



def _ollama_metadata_messages(page_text: str) -> list[dict[str, str]]:
    system_message = (
        "You extract bibliographic metadata from academic paper first-page text. "
        "Return JSON only. Do not invent missing values."
    )
    user_message = f"""
Text from the first pages of a PDF:
{collapse_whitespace(page_text)[:12000]}

Task:
Extract bibliographic metadata for this paper. Use only the text above.
Prefer the paper title, the paper authors, the publication year, and DOI if
visible. Do not use journal headers, page headers, copyright boilerplate,
section headings, Zotero storage identifiers, filenames, or abstracts as titles
or authors.

Return exactly this JSON shape:
{{
  "title": "... or null",
  "authors": "comma-separated author names, or null",
  "year": 2006,
  "doi": "10.xxxx/... or null",
  "confidence": "high|medium|low",
  "reason": "short explanation"
}}
""".strip()
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]



def _ollama_tags_url(ollama_url: str) -> str:
    return f"{ollama_url.rstrip('/')}/api/tags"


def _ollama_is_reachable(ollama_url: str, timeout_seconds: float = 1.0) -> bool:
    request = Request(
        _ollama_tags_url(ollama_url),
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except (OSError, URLError, TimeoutError):
        return False


def ensure_ollama_available(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    startup_timeout_seconds: float = 20.0,
    progress_callback: ProgressCallback | None = None,
) -> bool:
    """Ensure the Ollama HTTP server is reachable, starting it if needed.

    Returns True if this function started ``ollama serve``. Raises
    MetadataRefreshError if Ollama cannot be reached or started.
    """

    if _ollama_is_reachable(ollama_url):
        return False

    ollama_executable = shutil.which("ollama")

    if ollama_executable is None:
        raise MetadataRefreshError(
            "Ollama is not reachable, and the 'ollama' command was not found on PATH. "
            "Start Ollama manually or install the Ollama CLI."
        )

    if progress_callback is not None:
        progress_callback("Ollama is not reachable; starting 'ollama serve'...")

    try:
        subprocess.Popen(
            [ollama_executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise MetadataRefreshError(
            f"Ollama is not reachable, and starting 'ollama serve' failed: {exc}"
        ) from exc

    deadline = time.monotonic() + startup_timeout_seconds

    while time.monotonic() < deadline:
        if _ollama_is_reachable(ollama_url):
            return True
        time.sleep(0.25)

    raise MetadataRefreshError(
        "Started 'ollama serve', but Ollama did not become reachable at "
        f"{ollama_url!r} within {startup_timeout_seconds:g} seconds."
    )


def _call_ollama_json(
    messages: list[dict[str, str]],
    model: str,
    ollama_url: str,
    timeout_seconds: float,
) -> dict:
    api_url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 300},
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
        raise MetadataRefreshError("Ollama returned empty metadata JSON.")

    return json.loads(content)



def _int_or_none(value: object) -> int | None:
    if value is None:
        return None

    try:
        year = int(value)
    except (TypeError, ValueError):
        return None

    if _looks_like_bad_year(year):
        return None

    return year



def _metadata_from_ollama_payload(payload: dict) -> tuple[ExtractedMetadata, str, str]:
    title = clean_title_metadata_text(payload.get("title"))
    authors = clean_author_metadata_text(payload.get("authors"))
    doi = clean_metadata_text(payload.get("doi"))
    year = _int_or_none(payload.get("year"))
    confidence = clean_metadata_text(payload.get("confidence")) or "low"
    reason = clean_metadata_text(payload.get("reason")) or "Ollama metadata extraction."

    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    if title is not None and looks_like_bad_title(title):
        title = None

    if authors is not None and _looks_like_bad_authors(authors):
        authors = None

    if doi is not None and not DOI_PREFIX_RE.match(doi):
        doi = None

    return ExtractedMetadata(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
    ), confidence, reason



def extract_metadata_with_ollama(
    pdf_path: str | Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 60.0,
    max_pages: int = 3,
) -> MetadataRefreshProposal:
    """Ask Ollama to infer bibliographic metadata from early PDF text."""

    page_text = extract_text_from_first_pages(pdf_path, max_pages=max_pages)

    if not page_text.strip():
        return MetadataRefreshProposal(
            metadata=ExtractedMetadata(),
            source="llm",
            confidence="low",
            reason="No extractable text found in early PDF pages.",
        )

    try:
        payload = _call_ollama_json(
            _ollama_metadata_messages(page_text),
            model=model,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError, MetadataRefreshError) as exc:
        raise MetadataRefreshError(
            f"Ollama metadata extraction failed: {type(exc).__name__}: {exc}"
        ) from exc

    metadata, confidence, reason = _metadata_from_ollama_payload(payload)

    return MetadataRefreshProposal(
        metadata=metadata,
        source="llm",
        confidence=confidence,
        reason=reason,
    )



def _crossref_proposal(
    document: Document,
    early_text: str,
    crossref_mailto: str | None,
    timeout_seconds: float,
) -> MetadataRefreshProposal | None:
    doi = _best_doi_for_document(document, early_text)

    if doi is None:
        return None

    metadata = lookup_crossref_metadata(
        doi,
        crossref_mailto=crossref_mailto,
        timeout=timeout_seconds,
    )

    if metadata.doi is None:
        metadata = ExtractedMetadata(
            title=metadata.title,
            authors=clean_author_metadata_text(metadata.authors),
            year=metadata.year,
            doi=doi,
        )

    if not any([metadata.title, metadata.authors, metadata.year, metadata.doi]):
        return None

    confidence = "high" if _metadata_has_core_fields(metadata) else "medium"

    return MetadataRefreshProposal(
        metadata=metadata,
        source="crossref",
        confidence=confidence,
        reason=f"DOI resolved through Crossref: {doi}",
    )



def propose_metadata_refresh(
    document: Document,
    method: RefreshMethod = "auto",
    crossref_mailto: str | None = None,
    crossref_timeout_seconds: float = 10.0,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_timeout_seconds: float = 60.0,
    max_pages: int = 3,
) -> MetadataRefreshProposal:
    """Propose improved metadata for one document."""

    if method not in {"auto", "crossref", "llm"}:
        raise ValueError(f"Unknown metadata refresh method: {method}")

    early_text = extract_text_from_first_pages(document.pdf_path, max_pages=max_pages)

    if method in {"auto", "crossref"}:
        proposal = _crossref_proposal(
            document=document,
            early_text=early_text,
            crossref_mailto=crossref_mailto,
            timeout_seconds=crossref_timeout_seconds,
        )

        if proposal is not None and (
            method == "crossref" or _metadata_has_core_fields(proposal.metadata)
        ):
            return proposal

        if method == "crossref":
            return proposal or MetadataRefreshProposal(
                metadata=ExtractedMetadata(),
                source="none",
                confidence="low",
                reason="No DOI could be resolved through Crossref.",
            )

    if method in {"auto", "llm"}:
        return extract_metadata_with_ollama(
            document.pdf_path,
            model=ollama_model,
            ollama_url=ollama_url,
            timeout_seconds=ollama_timeout_seconds,
            max_pages=max_pages,
        )

    return MetadataRefreshProposal(
        metadata=ExtractedMetadata(),
        source="none",
        confidence="low",
        reason="No metadata refresh source was available.",
    )



def apply_metadata_refresh(
    document: Document,
    store,
    proposal: MetadataRefreshProposal,
    replace_all: bool = False,
) -> tuple[Document | None, dict[str, object]]:
    """Apply proposed metadata updates and return updated document plus updates."""

    updates = metadata_updates_for_document(
        document,
        proposal,
        replace_all=replace_all,
    )

    if not updates:
        return None, {}

    return store.update_document_metadata(document.doc_id, **updates), updates



def refresh_documents_metadata(
    documents: Iterable[Document],
    store,
    method: RefreshMethod = "auto",
    crossref_mailto: str | None = None,
    crossref_timeout_seconds: float = 10.0,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_timeout_seconds: float = 60.0,
    max_pages: int = 3,
    include_apparently_good: bool = False,
    apply: bool = False,
    replace_all_crossref: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[MetadataRefreshResult]:
    """Inspect and optionally refresh metadata for many documents."""

    results: list[MetadataRefreshResult] = []

    for document in documents:
        assessment = assess_document_metadata(document)

        if not assessment.needs_refresh and not include_apparently_good:
            results.append(
                MetadataRefreshResult(
                    document=document,
                    assessment=assessment,
                    proposal=None,
                    updates={},
                )
            )
            continue

        if progress_callback is not None:
            progress_callback(str(document.pdf_path))

        try:
            proposal = propose_metadata_refresh(
                document,
                method=method,
                crossref_mailto=crossref_mailto,
                crossref_timeout_seconds=crossref_timeout_seconds,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                ollama_timeout_seconds=ollama_timeout_seconds,
                max_pages=max_pages,
            )
            replace_all = replace_all_crossref and proposal.source == "crossref"
            updates = metadata_updates_for_document(
                document,
                proposal,
                replace_all=replace_all,
            )
            updated_document = None

            if apply and updates:
                updated_document = store.update_document_metadata(
                    document.doc_id,
                    **updates,
                )

            results.append(
                MetadataRefreshResult(
                    document=document,
                    assessment=assessment,
                    proposal=proposal,
                    updates=updates,
                    updated_document=updated_document,
                )
            )
        except Exception as exc:
            results.append(
                MetadataRefreshResult(
                    document=document,
                    assessment=assessment,
                    proposal=None,
                    updates={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return results
