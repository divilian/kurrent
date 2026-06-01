# Storage and retrieval layer for the kurrent corpus (documents, chunks, and
# vector search).

__all__ = [
    "get_document",
    "list_documents",
    "upsert_document",
    "upsert_chunks",
    "search",
]

def get_document(doc_id: str) -> DocumentRecord | None:
    """
    Used to display:

    title-ish label (maybe from filename)

    source paths

    ingest timestamp

    (later) Zotero metadata
    """

def list_documents(where: dict | None = None) -> list[DocumentRecord]:
    """
    Useful for:

    “show me what’s indexed”

    debugging

    future UI (“related papers” list)
    """

def upsert_document(doc: DocumentRecord) -> None:
    """
    Stores/updates the document metadata (doc_id, path(s), hashes, timestamps).

    Use case:

    you ingest A.pdf

    compute doc_id = sha256(bytes)

    record where you found it (source_path), file size/mtime, etc.

    “Upsert” here really means: doc_id is the primary key.
    """
    pass

def upsert_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    """
    Replaces (or updates) all chunks for a document.

    I strongly suggest the semantics be:

    delete existing chunks where doc_id == ..., then insert new ones

    because chunking strategies change over time.
    """
    pass

def search(
    query_embedding: np.ndarray,
    k: int,
    where: dict | None = None,
) -> list[SearchHit]:
    """
    Returns the top-k most similar chunks across the global corpus (or filtered
subset).

    Parquet backend: brute-force cosine in NumPy/Polars

    Deep Lake backend: dataset vector search / ANN index

    Return each hit with:

    similarity score

    chunk metadata + chunk_text (for prompt)

    doc_id (so you can group hits by paper)
    """
    pass

