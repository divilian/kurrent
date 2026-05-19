from pathlib import Path

import pymupdf
import pytest

from kurrent.chunker import chunker_version
from kurrent.embedder import Embedder
from kurrent.ingester import ingest_pdf
from kurrent.state_store import StateStore


MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] >>
endobj
trailer
<< /Root 1 0 R >>
%%EOF
"""


def write_pdf(path: Path, content: bytes = MINIMAL_PDF_BYTES) -> Path:
    path.write_bytes(content)
    return path

def write_text_pdf(path: Path, pages: list[str]) -> Path:
    pdf = pymupdf.open()

    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)

    pdf.save(path)
    pdf.close()

    return path


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "kurrent.db"
    with StateStore(db_path) as store:
        yield store

class FakeModel:
    def encode(self, texts, convert_to_numpy=True):
        return FakeEmbeddings([
            [float(len(text)), float(i), 1.0]
            for i, text in enumerate(texts)
        ])

class FakeEmbeddings:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values

@pytest.fixture
def embedder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kurrent.embedder.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    return Embedder(
        chroma_path=tmp_path / "chroma",
        model_name="fake-model",
    )

def test_ingest_pdf_creates_document(store, tmp_path):
    pdf_path = write_pdf(tmp_path / "paper.pdf")

    doc_id = ingest_pdf(pdf_path, store)

    doc = store.get_document(doc_id)

    assert doc is not None
    assert doc.doc_id == doc_id
    assert doc.pdf_path == pdf_path.resolve()
    assert doc.pdf_sha256 is not None
    assert doc.storage_mode == "external"
    assert doc.title is None
    assert doc.authors is None
    assert doc.year is None
    assert doc.doi is None


def test_ingest_same_pdf_twice_returns_same_doc_id(store, tmp_path):
    pdf_path = write_pdf(tmp_path / "paper.pdf")

    first_doc_id = ingest_pdf(pdf_path, store)
    second_doc_id = ingest_pdf(pdf_path, store)

    assert second_doc_id == first_doc_id


def test_ingest_same_pdf_contents_at_different_paths_returns_same_doc_id(
    store,
    tmp_path,
):
    first_path = write_pdf(tmp_path / "paper.pdf")
    second_path = write_pdf(tmp_path / "copy.pdf")

    first_doc_id = ingest_pdf(first_path, store)
    second_doc_id = ingest_pdf(second_path, store)

    assert second_doc_id == first_doc_id


def test_ingest_same_path_with_changed_contents_creates_new_document(
    store,
    tmp_path,
):
    pdf_path = write_pdf(tmp_path / "paper.pdf")

    first_doc_id = ingest_pdf(pdf_path, store)

    changed_pdf_bytes = MINIMAL_PDF_BYTES + b"\n% changed content\n"
    write_pdf(pdf_path, changed_pdf_bytes)

    second_doc_id = ingest_pdf(pdf_path, store)

    assert second_doc_id != first_doc_id


def test_ingest_non_pdf_raises_error(store, tmp_path):
    txt_path = tmp_path / "not-a-pdf.txt"
    txt_path.write_text("hello, not a PDF", encoding="utf-8")

    with pytest.raises(ValueError):
        ingest_pdf(txt_path, store)


def test_ingest_missing_file_raises_error(store, tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises((FileNotFoundError, ValueError)):
        ingest_pdf(missing_path, store)


def test_ingest_pdf_creates_chunks(store, tmp_path):
    pdf_path = write_text_pdf(
        tmp_path / "paper.pdf",
        ["This is page one."],
    )

    doc_id = ingest_pdf(pdf_path, store)

    chunks = store.get_chunks_for_document(
        doc_id=doc_id,
        chunker_version=chunker_version(),
    )

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk.doc_id == doc_id
    assert chunk.chunker_version == "word-aware-fixed-char-2000-v1"
    assert chunk.chunk_index == 0
    assert "This is page one." in chunk.text
    assert chunk.text_sha256 is not None
    assert chunk.page_start == 1
    assert chunk.page_end == 1


def test_ingest_pdf_with_embedder_indexes_chunks(
    store,
    embedder,
    tmp_path,
):
    pdf_path = write_text_pdf(
        tmp_path / "paper.pdf",
        ["This is page one."],
    )

    doc_id = ingest_pdf(pdf_path, store, embedder=embedder)

    chunks = store.get_chunks_for_document(
        doc_id=doc_id,
        chunker_version=chunker_version(),
    )

    assert len(chunks) == 1

    results = embedder.collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "embeddings"],
    )

    assert results["ids"] == [chunks[0].chunk_id]
    assert results["documents"] == ["This is page one."]

    metadata = results["metadatas"][0]
    assert metadata["doc_id"] == doc_id
    assert metadata["chunker_version"] == chunker_version()
    assert metadata["chunk_index"] == 0
    assert metadata["text_sha256"] == chunks[0].text_sha256
    assert metadata["embedding_model"] == "fake-model"
    assert metadata["page_start"] == 1
    assert metadata["page_end"] == 1

    assert list(results["embeddings"][0][:3]) == pytest.approx(
        [17.0, 0.0, 1.0]
    )


def test_ingest_pdf_with_embedder_is_idempotent(
    store,
    embedder,
    tmp_path,
):
    pdf_path = write_text_pdf(
        tmp_path / "paper.pdf",
        ["This is page one."],
    )

    first_doc_id = ingest_pdf(pdf_path, store, embedder=embedder)
    second_doc_id = ingest_pdf(pdf_path, store, embedder=embedder)

    assert second_doc_id == first_doc_id

    chunks = store.get_chunks_for_document(
        doc_id=first_doc_id,
        chunker_version=chunker_version(),
    )

    assert len(chunks) == 1

    results = embedder.collection.get(
        where={"doc_id": first_doc_id},
        include=["documents", "metadatas", "embeddings"],
    )

    assert len(results["ids"]) == 1
    assert results["ids"] == [chunks[0].chunk_id]
