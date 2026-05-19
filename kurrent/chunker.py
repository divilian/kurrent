# Functions to split text into retrievable chunks.
from pathlib import Path
import hashlib

import pymupdf

from kurrent.file_utils import is_pdf
from kurrent.state_store import StateStore
from kurrent.schema import Document, Chunk

__all__ = ["chunk_document"]


def chunker_version(target_chars: int = 2000) -> str:
    return f"word-aware-fixed-char-{target_chars}-v1"


def chunk_document(
    doc_id: str,
    store: StateStore,
) -> list[Chunk]:
    """
    Extract text from a stored PDF document, convert it into Chunk objects,
    store those chunks in kurrent state, and return them.
    """
    doc = store.get_document(doc_id)

    if doc is None:
        raise ValueError(f"No such document: {doc_id}")

    existing_chunks = store.get_chunks_for_document(
        doc_id=doc.doc_id,
        chunker_version=chunker_version(),
    )

    if existing_chunks:
        return existing_chunks

    pages = extract_pdf_pages(doc.pdf_path)
    chunks = make_word_aware_fixed_size_chunks(
        pages=pages,
        doc_id=doc.doc_id,
    )
    store.insert_chunks(chunks)
    return chunks


def make_word_aware_fixed_size_chunks(
    pages: dict[int, str],
    doc_id: str,
    # Roughly 300-500 English words. Enough to preserve scholarly content - a
    # full paragraph or several related paragraphs - but small enough that
    # embeddings are fairly focused.
    target_chars: int = 2000,
) -> list[Chunk]:
    """
    Convert page-indexed PDF text into word-aware, approximately fixed-size
    Chunk objects.

    The input pages maps 1-based page numbers to the extracted text for each
    page. Chunks are built by accumulating words until adding another word
    would exceed target_chars; chunk boundaries therefore avoid splitting
    words, but individual chunks may span page boundaries.

    Each returned Chunk records:
    - the supplied doc_id
    - a deterministic chunk_index based on chunk order
    - the generated chunker_version, including target_chars
    - the chunk text and its SHA256 hash
    - the first and last PDF page represented in the chunk

    The chunker version is generated internally as:

        word-aware-fixed-char-{target_chars}-v1

    This function only creates Chunk objects. It does not insert them into the
    database.
    """

    chunks: list[Chunk] = []

    current_parts: list[str] = []
    current_start_page: int | None = None
    current_end_page: int | None = None

    def current_text() -> str:
        return " ".join(current_parts).strip()

    def emit_chunk() -> None:
        text = current_text()
        if not text:
            return

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunker_version=chunker_version(target_chars),
                chunk_index=len(chunks),
                text=text,
                text_sha256=sha256_text(text),
                page_start=current_start_page,
                page_end=current_end_page,
            )
        )

    for page_num in sorted(pages):
        words = pages[page_num].split()

        for word in words:
            candidate_len = len(current_text()) + 1 + len(word)

            if current_parts and candidate_len > target_chars:
                emit_chunk()
                current_parts = []
                current_start_page = None
                current_end_page = None

            if current_start_page is None:
                current_start_page = page_num

            current_parts.append(word)
            current_end_page = page_num

    emit_chunk()

    return chunks


def extract_pdf_pages(path: str) -> dict[int, str]:
    """
    Given a path of a PDF file, return a dict whose ints are page numbers
    (1-based) and whose values are strings of text. Pages that are blank will
    still exist in the dict, with empty string as their value.
    """
    pages = {}
    with pymupdf.open(path) as pdf:
        # Use 1-based page numbers because that's what humans use.
        for page_num, page in enumerate(pdf, start=1):
            # Preserve empty pages.
            # (Note to later self: to get coordinates for auto-annotation, we
            # can instead use page.get_text("blocks") (or "words") here, which
            # return positional information.)
            pages[page_num] = page.get_text("text", sort=True) or ""

    return pages


def extract_pages(path: str) -> dict[int, str]:
    """
    (For now, we only ingest PDF files.)
    """
    if is_pdf(path):
        return extract_pdf_pages(path)
    raise ValueError(f"File {path} is not a valid PDF!")


def sha256_text(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":

    # Smoke test.
    from tempfile import TemporaryDirectory
    from pprint import pprint

    from kurrent.ingester import ingest_pdf

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "kurrent.db"

        with StateStore(db_path) as store:
            doc_id = ingest_pdf(
                "~/research/SSI/polarbear/CSSSA16/daviesZontine.pdf",
                store,
            )
            chunks = chunk_document(doc_id, store)

            for i, chunk in enumerate(chunks):
                print(f"Chunk {chunk.chunk_index}:")
                #chunk.text = chunk.text[:30] + "..." + chunk.text[-30:]
                pprint(chunk)

                if i < len(chunks) - 1:
                    input(f"Press Enter for chunk {chunks[i + 1].chunk_index}: ")
