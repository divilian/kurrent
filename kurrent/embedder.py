# Functions/classes to compute embeddings for chunks of text and index them in
# Chroma.
from __future__ import annotations
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import re
from typing import Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from kurrent.chunker import chunker_version
from kurrent.file_utils import normalize_path
from kurrent.pipeline import current_semantic_index_fingerprint
from kurrent.schema import Chunk, VectorChunkMatch
from kurrent.state_store import StateStore

__all__ = [
    "DEFAULT_EMBED_MODEL_NAME",
    "Embedder",
]

DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@contextmanager
def suppress_tqdm_for_model_loading():
    """Temporarily suppress dependency progress bars during model loading.

    SentenceTransformer/model-loading dependencies sometimes emit noisy tqdm
    bars such as "Loading weights" during CLI startup. Some of those bars do
    not honor TQDM_DISABLE reliably, so also silence stdout/stderr locally
    while the model object is constructed. This keeps kurrent's own
    user-facing tqdm progress bars during ingest/sectioning visible.

    Set KURRENT_SHOW_MODEL_LOAD_PROGRESS=1 to restore dependency model-load
    chatter for debugging.
    """

    previous = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"

    show_progress = os.environ.get("KURRENT_SHOW_MODEL_LOAD_PROGRESS") in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }

    try:
        if show_progress:
            yield
        else:
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
    finally:
        if previous is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = previous


class Embedder:
    """
    Coordinates embedding and indexing kurrent chunks into a Chroma collection.

    Note on the Chroma collection naming (and semantic index fingerprint):
    
    Kurrent uses a deliberately long Chroma collection name so that semantic
    search never mixes vectors produced by incompatible indexing pipelines.
    The name is a readable serialization of the exact pipeline that produced
    the chunk embeddings.
    
    Example:
    
      kurrent_chunks__index_semantic-index-fingerprint-v1_extractor_layout-aware-pymupdf-v2_sectioner_sectioner-v4_llm_sectioner_ollama-section-headings-v2_chunker_section-aware-fixed-char-v2_target_chars_2000__sentence-transformers_all-MiniLM-L6-v2
    
    Components:
    
    - kurrent_chunks__
      Fixed prefix. This collection stores Kurrent chunk embeddings.
    
    - index_semantic-index-fingerprint-v1
      Version of the semantic-index fingerprint scheme itself. If the recipe
      for deciding which pipeline components affect semantic compatibility
      changes, this version should change too.
    
    - extractor_layout-aware-pymupdf-v2
      PDF text extraction pipeline and version. Extraction changes can alter
      text order, page text, headings, and therefore downstream chunks.
    
    - sectioner_sectioner-v4
      Rules-based section detection pipeline and version. Section boundaries
      affect chunk boundaries and chunk metadata.
    
    - llm_sectioner_ollama-section-headings-v2
      LLM-assisted section heading detector and version. LLM heading choices
      can affect section boundaries, so they are part of the fingerprint.
    
    - chunker_section-aware-fixed-char-v2
      Chunking algorithm and version. If chunk construction, overlap, section
      handling, page provenance, or reference-section treatment changes, this
      should change.
    
    - target_chars_2000
      Chunker parameter. A different target chunk size produces different
      chunks and therefore different vectors.
    
    - sentence-transformers_all-MiniLM-L6-v2
      Embedding model used to generate vectors. Embeddings from different
      models are not comparable, so each model needs a distinct collection.
    
    In plain English, the full name means:
    
      "This collection contains Kurrent chunk vectors built from PDFs
      extracted with this extractor, sectioned with this sectioner, chunked
      with these chunking settings, and embedded with this embedding model."
    
    The name is ugly, but useful: if any semantically relevant part of the
    pipeline changes, Kurrent naturally uses a different Chroma collection
    rather than silently mixing stale and current vectors.
    """

    def __init__(
        self,
        chroma_path: str | Path,
        model_name: str = DEFAULT_EMBED_MODEL_NAME,
        collection_name: str | None = None,
    ):
        self.chroma_path = normalize_path(chroma_path)
        self.model_name = model_name
        self.semantic_index_fingerprint = current_semantic_index_fingerprint()
        self.collection_name = collection_name or self._make_collection_name(
            semantic_index_fingerprint=self.semantic_index_fingerprint,
            model_name=model_name,
        )

        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )

        with suppress_tqdm_for_model_loading():
            self.model = SentenceTransformer(model_name)

    def index_chunks(self, doc_id: str, store: StateStore) -> None:
        """
        Generate embeddings for this document's standard chunks and upsert them
        into the Chroma collection.

        Raises ValueError if the document has no chunks for the current standard
        chunker version.
        """
        chunks = store.get_chunks_for_document(
            doc_id=doc_id,
            chunker_version=chunker_version(),
        )

        if not chunks:
            raise ValueError(f"No chunks found for document: {doc_id}")

        pipeline_fingerprint = store.get_document_pipeline_fingerprint(doc_id)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                self._metadata_for_chunk(
                    chunk,
                    pipeline_fingerprint=pipeline_fingerprint,
                    semantic_index_fingerprint=self.semantic_index_fingerprint,
                )
                for chunk in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        """Remove all vector-index entries for one document from this collection."""

        self.collection.delete(
            where={"doc_id": doc_id},
        )

    def has_document(self, doc_id: str) -> bool:
        """Return whether this Chroma collection has entries for a document."""

        results = self.collection.get(
            where={"doc_id": doc_id},
            limit=1,
            include=[],
        )

        return bool(results.get("ids"))

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a list of texts using this Embedder's
        SentenceTransformers model.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def _make_collection_name(
        semantic_index_fingerprint: str,
        model_name: str,
    ) -> str:
        """
        Build a Chroma-safe collection name from the semantic index and
        embedding model.

        Chroma collection names cannot contain arbitrary characters such as
        slashes, so model names like sentence-transformers/all-MiniLM-L6-v2 and
        pipeline fingerprints are sanitized.
        """
        raw_name = f"kurrent_chunks__{semantic_index_fingerprint}__{model_name}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
        safe_name = re.sub(r"\.{2,}", ".", safe_name)
        safe_name = safe_name.strip("._-")

        if len(safe_name) < 3:
            raise ValueError(
                f"Generated Chroma collection name is too short: {safe_name}"
            )

        return safe_name[:512]

    def _metadata_for_chunk(
        self,
        chunk: Chunk,
        pipeline_fingerprint: str | None = None,
        semantic_index_fingerprint: str | None = None,
    ) -> dict:
        """
        Return Chroma metadata for a Chunk.

        Chroma metadata values should be scalar values, and None values are best
        avoided, so page_start/page_end and fingerprints are included only when
        present.
        """
        metadata = {
            "doc_id": chunk.doc_id,
            "chunker_version": chunk.chunker_version,
            "chunk_index": chunk.chunk_index,
            "text_sha256": chunk.text_sha256,
            "embedding_model": getattr(
                self,
                "model_name",
                DEFAULT_EMBED_MODEL_NAME,
            ),
        }

        if pipeline_fingerprint is not None:
            metadata["text_pipeline_fingerprint"] = pipeline_fingerprint

        if semantic_index_fingerprint is not None:
            metadata["semantic_index_fingerprint"] = semantic_index_fingerprint

        if chunk.page_start is not None:
            metadata["page_start"] = chunk.page_start

        if chunk.page_end is not None:
            metadata["page_end"] = chunk.page_end

        return metadata

    def query_chunks(
        self,
        search_text: str,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: Sequence[str] | None = None,
    ) -> list[VectorChunkMatch]:
        """Query Chroma for chunks semantically close to search_text."""

        query_embedding = self.generate_embeddings([search_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunk_ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        excluded = set(exclude_doc_ids or [])

        matches: list[VectorChunkMatch] = []

        for chunk_id, document, metadata, distance in zip(
            chunk_ids,
            documents,
            metadatas,
            distances,
        ):
            if max_distance is not None and distance > max_distance:
                continue

            doc_id = metadata.get("doc_id")
            if doc_id in excluded:
                continue

            matches.append(
                VectorChunkMatch(
                    chunk_id=chunk_id,
                    distance=distance,
                    text=document,
                )
            )

        return matches

    def query_similar_chunks_by_chunk_id(
        self,
        chunk_id: str,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: Sequence[str] | None = None,
    ) -> list[VectorChunkMatch]:
        """Query Chroma for chunks similar to an already-indexed chunk."""

        source = self.collection.get(
            ids=[chunk_id],
            include=["embeddings"],
        )

        if not source["ids"]:
            raise ValueError(f"Chunk not found in vector index: {chunk_id!r}")

        embedding = source["embeddings"][0]

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunk_ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        excluded = set(exclude_doc_ids or [])

        matches: list[VectorChunkMatch] = []

        for result_chunk_id, document, metadata, distance in zip(
            chunk_ids,
            documents,
            metadatas,
            distances,
        ):
            if max_distance is not None and distance > max_distance:
                continue

            doc_id = metadata.get("doc_id")
            if doc_id in excluded:
                continue

            matches.append(
                VectorChunkMatch(
                    chunk_id=result_chunk_id,
                    distance=distance,
                    text=document,
                )
            )

        return matches


if __name__ == "__main__":

    # Smoke test.
    from pathlib import Path
    from pprint import pprint

    from kurrent.ingester import ingest_pdf

    pdf_path = Path("/home/stephen/teaching/419/syllabus.pdf")

    tmpdir = Path("/tmp/embedder")
    if not tmpdir.is_dir():
        tmpdir.mkdir(parents=True)
    db_path = tmpdir / "kurrent.db"
    chroma_path = tmpdir / "chroma"

    with StateStore(db_path) as store:
        doc_id = ingest_pdf(pdf_path, store)

        embedder = Embedder(chroma_path)
        embedder.index_chunks(doc_id, store)

        results = embedder.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "embeddings"],
        )

        print(f"Indexed document: {doc_id}")
        print(f"Collection: {embedder.collection_name}")
        print(f"Indexed chunks: {len(results['ids'])}")

        for i, chunk_id in enumerate(results["ids"]):
            print(f"\nChunk ID: {chunk_id}")
            print("Metadata:")
            pprint(results["metadatas"][i])

            print("\nEmbedding preview:")
            embedding = results["embeddings"][i]
            print(embedding[:8])
            print(f"Embedding dimensions: {len(embedding)}")

            print("\nText preview:")
            print(results["documents"][i][:500])
