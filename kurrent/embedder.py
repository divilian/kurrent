# Functions/classes to compute embeddings for chunks of text and index them in
# Chroma.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from kurrent.chunker import chunker_version
from kurrent.file_utils import normalize_path
from kurrent.schema import Chunk, VectorChunkMatch, make_chunk_id
from kurrent.state_store import StateStore


DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """
    Coordinates embedding and indexing kurrent chunks into a Chroma collection.
    """

    def __init__(
        self,
        chroma_path: str | Path,
        model_name: str = DEFAULT_EMBED_MODEL_NAME,
        collection_name: str | None = None,
    ):
        self.chroma_path = normalize_path(chroma_path)
        self.model_name = model_name
        self.collection_name = collection_name or self._make_collection_name(
            chunker_version=chunker_version(),
            model_name=model_name,
        )

        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )

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

        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                self._metadata_for_chunk(chunk)
                for chunk in chunks
            ],
        )

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a list of texts using this Embedder's
        SentenceTransformers model.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def _make_collection_name(
        chunker_version: str,
        model_name: str,
    ) -> str:
        """
        Build a Chroma-safe collection name from the chunker and embedding
        model.

        Chroma collection names cannot contain arbitrary characters such as
        slashes, so model names like sentence-transformers/all-MiniLM-L6-v2 are
        sanitized.
        """
        raw_name = f"kurrent_chunks__{chunker_version}__{model_name}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
        safe_name = re.sub(r"\.{2,}", ".", safe_name)
        safe_name = safe_name.strip("._-")

        if len(safe_name) < 3:
            raise ValueError(
                f"Generated Chroma collection name is too short: {safe_name}"
            )

        return safe_name[:512]

    def _metadata_for_chunk(self, chunk: Chunk) -> dict:
        """
        Return Chroma metadata for a Chunk.

        Chroma metadata values should be scalar values, and None values are best
        avoided, so page_start/page_end are included only when present.
        """
        metadata = {
            "doc_id": chunk.doc_id,
            "chunker_version": chunk.chunker_version,
            "chunk_index": chunk.chunk_index,
            "text_sha256": chunk.text_sha256,
            "embedding_model": self.model_name,
        }

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

        results = self.collection.query(
            query_texts=[search_text],
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

if __name__ == "__main__":

    # Smoke test.
    from pathlib import Path
    from pprint import pprint
    from tempfile import TemporaryDirectory

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
