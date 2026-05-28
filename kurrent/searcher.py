"""User-facing search orchestration for kurrent.

The Searcher coordinates between the vector index, the kurrent state database,
and later full-text search machinery.

Terminology:
- search_* methods are user-facing workflows.
- query_* methods live on lower-level backends such as Embedder and StateStore.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from kurrent.schema import ChunkHit, DocumentHit
from kurrent.state_store import StateStore
from kurrent.sectioner import is_reference_section_chunk

if TYPE_CHECKING:
    from kurrent.embedder import Embedder


class Searcher:
    """Coordinate user-facing search workflows."""

    def __init__(
        self,
        state_store: StateStore,
        embedder: Embedder | None = None,
    ) -> None:
        self.state_store = state_store
        self.embedder = embedder

    def semantic_chunk_search(
        self,
        search_text: str,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: Sequence[str] | None = None,
        include_reference_sections: bool = False,
    ) -> list[ChunkHit]:
        """Find chunks semantically similar to a free-text search expression.

        By default, chunks from reference/bibliography sections are excluded
        because they often create high-vocabulary false positives. Pass
        include_reference_sections=True to include them.
        """
        if self.embedder is None:
            raise ValueError("Semantic search requires an Embedder.")

        vector_matches = self.embedder.query_chunks(
            search_text,
            n_results=n_results,
            max_distance=max_distance,
            exclude_doc_ids=exclude_doc_ids,
        )

        hits: list[ChunkHit] = []

        for match in vector_matches:
            chunk = self.state_store.get_chunk(match.chunk_id)

            if chunk is None:
                raise ValueError(
                    "Vector index returned a chunk_id not found in "
                    f"kurrent state: {match.chunk_id!r}"
                )

            if (
                not include_reference_sections
                and is_reference_section_chunk(chunk)
            ):
                continue

            document = self.state_store.get_document(chunk.doc_id)

            if document is None:
                raise ValueError(
                    "Chunk exists in kurrent state, but parent document is "
                    f"missing: {chunk.doc_id!r}"
                )

            hits.append(
                ChunkHit(
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
            )

        return hits

    def semantic_document_search(
        self,
        search_text: str,
        max_documents: int = 10,
        max_distance: float | None = None,
        include_reference_sections: bool = False,
    ) -> list[DocumentHit]:
        """Find documents by aggregating semantic chunk search results.

        This searches chunks first, groups matching chunks by parent document,
        and ranks documents by each document's best chunk distance.
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

            if hit.distance is None:
                continue

            if curr_best.distance is None or hit.distance < curr_best.distance:
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
                    score=best_hit.distance,
                    best_chunk_id=best_hit.chunk_id,
                )
            )

        document_hits.sort(
            key=lambda hit: float("inf") if hit.score is None else hit.score
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

    return {
        "store": store,
        "embedder": embedder,
        "searcher": searcher,
        "doc_id": doc_id,
        "search_text": search_text,
        "hits": hits,
        "smoke_dir": smoke_dir,
        "db_path": db_path,
        "chroma_path": chroma_path,
        "pdf_path": pdf_path,
    }


def print_smoke_summary(ns: dict) -> None:
    """Print a readable summary of the smoke-test objects."""

    smoke_dir = ns["smoke_dir"]
    db_path = ns["db_path"]
    chroma_path = ns["chroma_path"]
    pdf_path = ns["pdf_path"]
    doc_id = ns["doc_id"]
    search_text = ns["search_text"]
    hits = ns["hits"]

    print()
    print(f"Smoke directory: {smoke_dir}")
    print(f"Database path:   {db_path}")
    print(f"Chroma path:     {chroma_path}")
    print(f"PDF path:        {pdf_path}")
    print(f"Document ID:     {doc_id}")
    print(f"Search text:     {search_text!r}")
    print(f"Hits returned:   {len(hits)}")
    print()


if __name__ == "__main__":

    # Smoke test / IPython playground.
    #
    # Run from IPython with:
    #
    #     %run -i -m kurrent.searcher
    #
    # Then inspect:
    #
    #     searcher
    #     hits
    #     searcher.semantic_chunk_search("grading policy")

    from pprint import pprint

    from kurrent.embedder import Embedder
    from kurrent.ingester import ingest_pdf
    from kurrent.state_store import StateStore

    pdf_path = Path("/home/stephen/teaching/419/syllabus.pdf")

    tmpdir = Path("/tmp/searcher")
    if not tmpdir.is_dir():
        tmpdir.mkdir(parents=True)

    db_path = tmpdir / "kurrent.db"
    chroma_path = tmpdir / "chroma"

    # Deliberately not using a context manager here, so the store remains open
    # after %run finishes and searcher remains usable from IPython.
    store = StateStore(db_path)

    doc_id = ingest_pdf(pdf_path, store)

    embedder = Embedder(chroma_path)
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

    print(f"Indexed document: {doc_id}")
    print(f"Collection: {embedder.collection_name}")
    print(f"Search text: {search_text!r}")
    print(f"Hits: {len(hits)}")

    for i, hit in enumerate(hits, start=1):
        print(f"\nHit {i}")
        print(f"Chunk ID: {hit.chunk_id}")
        print(f"Distance: {hit.distance}")
        print(f"Path: {hit.path}")
        print(f"Title: {hit.title}")
        print(f"Pages: {hit.page_start}–{hit.page_end}")
        print("\nText preview:")
        print(hit.text[:500])
