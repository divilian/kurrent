"""User-facing search orchestration for kurrent.

The Searcher coordinates between the vector index, the kurrent state database,
and later full-text search machinery.

Terminology:
- search_* methods are user-facing workflows.
- query_* methods live on lower-level backends such as Embedder and StateStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING, Sequence

from kurrent.chunker import chunker_version
from kurrent.schema import ChunkHit, DocumentHit
from kurrent.state_store import StateStore
from kurrent.sectioner import is_reference_section_chunk

__all__ = [
    "Searcher",
    "make_smoke_searcher",
    "print_smoke_summary",
]

if TYPE_CHECKING:
    from kurrent.embedder import Embedder


# Keep this deliberately small. The goal is not linguistic perfection; it is to
# keep high-frequency glue words from dominating lexical rescue for queries like
# "a personal knowledge base" while preserving meaningful short terms such as
# acronyms (PKB), model names, and author names.
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "using",
    "with",
}


@dataclass(slots=True)
class _HybridCandidate:
    """Internal candidate used while merging retrieval signals."""

    hit: ChunkHit
    semantic_distance: float | None = None
    from_semantic: bool = False
    from_lexical: bool = False
    from_metadata: bool = False
    reasons: set[str] = field(default_factory=set)


class Searcher:
    """Coordinate user-facing search workflows."""

    def __init__(
        self,
        state_store: StateStore,
        embedder: Embedder | None = None,
    ) -> None:
        self.state_store = state_store
        self.embedder = embedder

    @staticmethod
    def _query_terms(search_text: str) -> list[str]:
        """Return normalized non-stopword query terms for lexical rescue."""

        terms: list[str] = []
        seen = set()

        for raw_term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", search_text):
            term = raw_term.strip("_'-").lower()

            if len(term) < 2:
                continue

            if term in _QUERY_STOPWORDS:
                continue

            if term in seen:
                continue

            seen.add(term)
            terms.append(term)

        return terms

    @staticmethod
    def _normalized_phrase(search_text: str) -> str:
        return " ".join(search_text.lower().split())

    @staticmethod
    def _metadata_text(document) -> str:
        values = [
            getattr(document, "title", None),
            getattr(document, "authors", None),
            getattr(document, "year", None),
            getattr(document, "doi", None),
            getattr(document, "pdf_path", None),
        ]
        return " ".join(str(value) for value in values if value is not None).lower()

    @staticmethod
    def _semantic_score(distance: float | None) -> float:
        """Convert a lower-is-better vector distance into higher-is-better score."""

        if distance is None:
            return 0.0

        # Chroma cosine distances are usually near 0 for very close matches and
        # increase as matches worsen. Clamp so odd/index-dependent distances do
        # not create large negative scores.
        return max(0.0, 1.0 - float(distance))

    @classmethod
    def _lexical_score_for_text(
        cls,
        text: str,
        search_text: str,
        terms: Sequence[str],
    ) -> float:
        """Return a lexical evidence score for one chunk/section text."""

        normalized_text = " ".join(text.lower().split())

        if not normalized_text:
            return 0.0

        score = 0.0
        phrase = cls._normalized_phrase(search_text)

        if phrase and len(phrase) >= 4 and phrase in normalized_text:
            score += 2.0

        if terms:
            matched_terms = [term for term in terms if term in normalized_text]
            coverage = len(matched_terms) / len(terms)
            score += 1.2 * coverage
            score += 0.1 * min(len(matched_terms), 5)

        return score

    @classmethod
    def _metadata_score_for_document(
        cls,
        document,
        search_text: str,
        terms: Sequence[str],
    ) -> float:
        """Return a metadata evidence score for one document."""

        metadata_text = cls._metadata_text(document)

        if not metadata_text:
            return 0.0

        score = 0.0
        phrase = cls._normalized_phrase(search_text)

        if phrase and len(phrase) >= 4 and phrase in metadata_text:
            score += 1.75

        if terms:
            matched_terms = [term for term in terms if term in metadata_text]
            coverage = len(matched_terms) / len(terms)
            score += 1.4 * coverage
            score += 0.15 * min(len(matched_terms), 5)

        return score

    def _chunk_hit_from_vector_match(self, match) -> ChunkHit:
        chunk = self.state_store.get_chunk(match.chunk_id)

        if chunk is None:
            raise ValueError(
                "Vector index returned a chunk_id not found in "
                f"kurrent state: {match.chunk_id!r}"
            )

        document = self.state_store.get_document(chunk.doc_id)

        if document is None:
            raise ValueError(
                "Chunk exists in kurrent state, but parent document is "
                f"missing: {chunk.doc_id!r}"
            )

        return ChunkHit(
            chunk_id=match.chunk_id,
            distance=match.distance,
            text=chunk.text,  # SQLite version of text is authoritative
            path=document.pdf_path,
            title=document.title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_index=chunk.section_index,
            section_number=chunk.section_number,
            section_title=chunk.section_title,
        )

    def _add_candidate(
        self,
        candidates: dict[str, _HybridCandidate],
        hit: ChunkHit,
        *,
        source: str,
        semantic_distance: float | None = None,
    ) -> None:
        candidate = candidates.get(hit.chunk_id)

        if candidate is None:
            candidate = _HybridCandidate(hit=hit)
            candidates[hit.chunk_id] = candidate

        if semantic_distance is not None:
            candidate.semantic_distance = semantic_distance
            candidate.hit = ChunkHit(
                chunk_id=hit.chunk_id,
                distance=semantic_distance,
                text=hit.text,
                path=hit.path,
                title=hit.title,
                page_start=hit.page_start,
                page_end=hit.page_end,
                section_index=hit.section_index,
                section_number=hit.section_number,
                section_title=hit.section_title,
                score=hit.score,
                match_reasons=hit.match_reasons,
            )

        if source == "semantic":
            candidate.from_semantic = True
            candidate.reasons.add("semantic")
        elif source == "lexical":
            candidate.from_lexical = True
            candidate.reasons.add("lexical")
        elif source == "metadata":
            candidate.from_metadata = True
            candidate.reasons.add("metadata")
        else:
            candidate.reasons.add(source)

    def _rank_hybrid_candidates(
        self,
        candidates: dict[str, _HybridCandidate],
        search_text: str,
        terms: Sequence[str],
        include_reference_sections: bool,
    ) -> list[ChunkHit]:
        scored_hits: list[tuple[float, float, int, ChunkHit]] = []

        for candidate in candidates.values():
            hit = candidate.hit
            chunk = self.state_store.get_chunk(hit.chunk_id)

            if chunk is None:
                continue

            if not include_reference_sections and is_reference_section_chunk(chunk):
                continue

            document = self.state_store.get_document(hit.doc_id)

            if document is None:
                continue

            semantic = self._semantic_score(candidate.semantic_distance)
            lexical = self._lexical_score_for_text(
                "\n".join(
                    part
                    for part in [hit.section_title, hit.text]
                    if part
                ),
                search_text,
                terms,
            )
            metadata = self._metadata_score_for_document(
                document,
                search_text,
                terms,
            )

            # Semantic similarity remains the base signal. Lexical and metadata
            # evidence can rescue candidates Chroma missed, and can promote
            # exact concept/metadata matches over merely adjacent semantic hits.
            score = semantic + lexical + metadata

            if is_reference_section_chunk(chunk):
                score -= 0.5

            reasons = set(candidate.reasons)

            if lexical > 0:
                reasons.add("lexical")
            if metadata > 0:
                reasons.add("metadata")

            ranked_hit = ChunkHit(
                chunk_id=hit.chunk_id,
                distance=candidate.semantic_distance,
                text=hit.text,
                path=hit.path,
                title=hit.title,
                page_start=hit.page_start,
                page_end=hit.page_end,
                section_index=hit.section_index,
                section_number=hit.section_number,
                section_title=hit.section_title,
                score=score,
                match_reasons=tuple(sorted(reasons)),
            )
            distance_sort = (
                float("inf")
                if candidate.semantic_distance is None
                else candidate.semantic_distance
            )
            scored_hits.append(
                (
                    -score,
                    distance_sort,
                    hit.chunk_index,
                    ranked_hit,
                )
            )

        scored_hits.sort(key=lambda item: item[:3])
        return [item[3] for item in scored_hits]

    def semantic_chunk_search(
        self,
        search_text: str,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: Sequence[str] | None = None,
        include_reference_sections: bool = False,
    ) -> list[ChunkHit]:
        """Find chunks relevant to a free-text search expression.

        Retrieval is hybrid by default: Kurrent gathers semantic candidates from
        Chroma, lexical candidates from SQLite chunk text, and metadata-rescue
        candidates from SQLite document metadata. The merged pool is reranked
        with a simple higher-is-better score while preserving raw vector
        distances on hits that came from Chroma.
        """
        if self.embedder is None:
            raise ValueError("Semantic search requires an Embedder.")

        excluded = set(exclude_doc_ids or [])
        terms = self._query_terms(search_text)
        candidates: dict[str, _HybridCandidate] = {}

        semantic_candidate_count = max(n_results * 5, 40)
        lexical_candidate_count = max(n_results * 5, 40)
        metadata_candidate_count = max(n_results * 3, 20)

        vector_matches = self.embedder.query_chunks(
            search_text,
            n_results=semantic_candidate_count,
            max_distance=max_distance,
            exclude_doc_ids=exclude_doc_ids,
        )

        for match in vector_matches:
            hit = self._chunk_hit_from_vector_match(match)
            self._add_candidate(
                candidates,
                hit,
                source="semantic",
                semantic_distance=match.distance,
            )

        lexical_hits: list[ChunkHit] = []
        exact_hits = self.state_store.search_chunks_by_fulltext(
            search_text,
            limit=lexical_candidate_count,
        )
        lexical_hits.extend(exact_hits)

        if terms:
            lexical_hits.extend(
                self.state_store.search_chunks_by_any_terms(
                    list(terms),
                    limit=lexical_candidate_count * 5,
                    chunker_version=chunker_version(),
                )
            )

        for hit in lexical_hits:
            if hit.doc_id in excluded:
                continue
            self._add_candidate(candidates, hit, source="lexical")

        metadata_documents = self.state_store.search_documents_by_metadata(
            search_text,
            limit=metadata_candidate_count,
        )

        if terms:
            metadata_documents.extend(
                self.state_store.search_documents_by_metadata_terms(
                    list(terms),
                    limit=metadata_candidate_count,
                )
            )

        metadata_doc_ids: list[str] = []
        seen_metadata_doc_ids = set()

        for document in metadata_documents:
            if document.doc_id in excluded:
                continue
            if document.doc_id in seen_metadata_doc_ids:
                continue
            seen_metadata_doc_ids.add(document.doc_id)
            metadata_doc_ids.append(document.doc_id)

        metadata_hits = self.state_store.get_initial_chunks_for_documents(
            metadata_doc_ids,
            chunker_version=chunker_version(),
            chunks_per_document=2,
        )

        for hit in metadata_hits:
            self._add_candidate(candidates, hit, source="metadata")

        ranked_hits = self._rank_hybrid_candidates(
            candidates,
            search_text=search_text,
            terms=terms,
            include_reference_sections=include_reference_sections,
        )

        return ranked_hits[:n_results]

    @staticmethod
    def _hit_rank_score(hit: ChunkHit) -> float:
        if hit.score is not None:
            return hit.score

        if hit.distance is not None:
            return max(0.0, 1.0 - hit.distance)

        return 0.0

    def semantic_document_search(
        self,
        search_text: str,
        max_documents: int = 10,
        max_distance: float | None = None,
        include_reference_sections: bool = False,
    ) -> list[DocumentHit]:
        """Find documents by aggregating semantic chunk search results.

        This searches chunks first, groups matching chunks by parent document,
        and ranks documents by each document's best hybrid chunk score.
        """
        # A best guess heuristic.
        chunk_results = max(25, max_documents * 5)

        chunk_hits = self.semantic_chunk_search(
            search_text,
            n_results=chunk_results,
            max_distance=max_distance,
            include_reference_sections=include_reference_sections,
        )

        best_hit_by_doc_id: dict[str, ChunkHit] = {}

        for hit in chunk_hits:
            curr_best = best_hit_by_doc_id.get(hit.doc_id)

            if curr_best is None:
                best_hit_by_doc_id[hit.doc_id] = hit
                continue

            if self._hit_rank_score(hit) > self._hit_rank_score(curr_best):
                best_hit_by_doc_id[hit.doc_id] = hit

        document_hits: list[DocumentHit] = []

        for doc_id, best_hit in best_hit_by_doc_id.items():
            document = self.state_store.get_document(doc_id)

            if document is None:
                raise ValueError(
                    "Chunk search returned a hit whose parent document is "
                    f"missing from kurrent state: {doc_id!r}"
                )

            document_hits.append(
                DocumentHit(
                    doc_id=doc_id,
                    path=document.pdf_path,
                    title=document.title,
                    authors=document.authors,
                    year=document.year,
                    score=self._hit_rank_score(best_hit),
                    best_chunk_id=best_hit.chunk_id,
                )
            )

        document_hits.sort(
            key=lambda hit: float("-inf") if hit.score is None else -hit.score
        )

        return document_hits[:max_documents]

    def metadata_search(
        self,
        search_text: str,
        limit: int = 50,
    ) -> list[DocumentHit]:
        """Find documents whose metadata contains the search text.

        Metadata search checks title, authors, year, DOI, and PDF path. It is
        intentionally a hard-edged SQLite search rather than a semantic search.
        """
        documents = self.state_store.search_documents_by_metadata(
            search_text,
            limit=limit,
        )

        return [
            DocumentHit(
                doc_id=document.doc_id,
                path=document.pdf_path,
                title=document.title,
                authors=document.authors,
                year=document.year,
                score=None,
                best_chunk_id=None,
            )
            for document in documents
        ]

    def full_text_search(
        self,
        search_text: str,
        limit: int = 50,
    ) -> list[ChunkHit]:
        """Find chunks whose stored text contains the search text.

        This is lexical/substring search over chunk text, backed by SQLite LIKE
        in StateStore. It does not use embeddings.
        """
        return self.state_store.search_chunks_by_fulltext(
            search_text,
            limit=limit,
        )



def make_smoke_searcher() -> dict:
    """Build a persistent searcher smoke-test playground.

    This is intended for manual development, not unit testing.
    It creates live objects and returns them for inspection.
    """
    from pathlib import Path
    import shutil

    from kurrent.embedder import Embedder
    from kurrent.ingester import ingest_pdf
    from kurrent.state_store import StateStore

    pdf_path = Path("/home/stephen/teaching/419/syllabus.pdf")

    smoke_dir = Path.home() / "tmp" / "kurrent-smoke" / "searcher"
    db_path = smoke_dir / "kurrent.db"
    chroma_path = smoke_dir / "chroma"

    reset = False

    if reset and smoke_dir.exists():
        shutil.rmtree(smoke_dir)

    smoke_dir.mkdir(parents=True, exist_ok=True)
    store = StateStore(db_path)

    doc_id = ingest_pdf(pdf_path, store)

    embedder = Embedder(chroma_path=chroma_path)
    embedder.index_chunks(doc_id, store)

    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    search_text = "course policies and assignments"

    hits = searcher.semantic_chunk_search(
        search_text,
        n_results=5,
    )

    print_smoke_summary(search_text, hits)

    return {
        "searcher": searcher,
        "store": store,
        "embedder": embedder,
        "doc_id": doc_id,
        "hits": hits,
    }



def print_smoke_summary(search_text: str, hits: list[ChunkHit]) -> None:
    """Print a tiny summary for manual smoke testing."""
    print(f"Semantic search: {search_text!r}")
    print(f"Hits: {len(hits)}")

    for i, hit in enumerate(hits, start=1):
        print(f"\n{i}. {hit.title or hit.path}")
        print(f"Distance: {hit.distance}")
        print(f"Score: {hit.score}")
        print(hit.text[:500])
