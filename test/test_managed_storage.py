from pathlib import Path

import pytest

from kurrent import cli
from kurrent.config import get_kurrent_state_paths
from kurrent.file_utils import sha256_file
from kurrent.pdf_store import copy_pdf_to_managed_store, managed_pdf_filename
from kurrent.schema import Document, ExtractedMetadata


def write_fake_pdf(path: Path, body: bytes = b"fake test pdf\n") -> Path:
    """Write a minimal file that kurrent.file_utils.is_pdf accepts."""

    path.write_bytes(b"%PDF-" + body)
    return path


class FakeStore:
    """Small in-memory stand-in for the StateStore methods ingest uses."""

    def __init__(self) -> None:
        self.documents_by_sha256 = {}
        self.inserted_documents = []
        self.updated_metadata = []
        self.current_chunks = {}

    def get_document_by_sha256(self, pdf_sha256: str):
        return self.documents_by_sha256.get(pdf_sha256)

    def insert_document(self, document) -> None:
        self.documents_by_sha256[document.pdf_sha256] = document
        self.inserted_documents.append(document)

    def update_document_metadata(self, doc_id: str, **updates):
        self.updated_metadata.append((doc_id, updates))
        return self.inserted_documents[-1]

    def get_chunks_for_document(self, doc_id: str, chunker_version: str):
        return self.current_chunks.get((doc_id, chunker_version), [])


class FakeEmbedder:
    """Small stand-in for Embedder.index_chunks()."""

    def __init__(self) -> None:
        self.indexed_doc_ids = []

    def index_chunks(self, doc_id: str, store) -> None:
        self.indexed_doc_ids.append(doc_id)


def patch_chunking(monkeypatch) -> None:
    """Replace chunk_document() so ingest tests do not parse real PDFs."""

    def fake_chunk_document(doc_id, store, **kwargs) -> None:
        return None

    monkeypatch.setattr("kurrent.chunker.chunk_document", fake_chunk_document)


def test_state_paths_include_managed_pdfs_directory(tmp_path):
    """Verify that kurrent state now has a sibling pdfs/ directory path."""

    paths = get_kurrent_state_paths(tmp_path)

    assert paths.state_dir == tmp_path.resolve()
    assert paths.sqlite_path == tmp_path.resolve() / "kurrent.db"
    assert paths.chroma_path == tmp_path.resolve() / "chroma"
    assert paths.pdfs_path == tmp_path.resolve() / "pdfs"


def test_ingest_defaults_to_managed_storage_mode():
    """Verify that ingest defaults to managed PDF storage."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "paper.pdf"])

    assert args.command == "ingest"
    assert args.in_place is False


def test_ingest_accepts_in_place_storage_flag():
    """Verify that --in-place selects external/in-place PDF storage."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--in-place", "paper.pdf"])

    assert args.in_place is True


def test_ingest_accepts_external_as_alias_for_in_place():
    """Verify that --external remains an accepted alias for --in-place."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--external", "paper.pdf"])

    assert args.in_place is True


def test_managed_pdf_filename_is_readable_and_hash_disambiguated():
    """Verify that managed filenames keep a readable stem plus short hash."""

    filename = managed_pdf_filename(
        Path("Betz 2022 Final!.pdf"),
        "a3f8bb3bc54a4983af6318d0404e2d75",
    )

    assert filename.startswith("Betz-2022-Final")
    assert filename.endswith("--a3f8bb3bc54a.pdf")
    assert " " not in filename
    assert "!" not in filename


def test_managed_pdf_filename_disambiguates_same_named_files_by_hash():
    """Verify that same-named PDFs with different contents get different names."""

    first = managed_pdf_filename(Path("paper.pdf"), "a" * 64)
    second = managed_pdf_filename(Path("paper.pdf"), "b" * 64)

    assert first == "paper--aaaaaaaaaaaa.pdf"
    assert second == "paper--bbbbbbbbbbbb.pdf"
    assert first != second


def test_copy_pdf_to_managed_store_creates_directory_and_preserves_bytes(tmp_path):
    """Verify that managed storage creates pdfs/ and copies bytes exactly."""

    source_pdf = write_fake_pdf(tmp_path / "Source Paper.pdf", b"managed bytes\n")
    pdf_sha256 = sha256_file(source_pdf)
    pdfs_dir = tmp_path / "state" / "pdfs"

    managed_path = copy_pdf_to_managed_store(
        source_path=source_pdf,
        pdfs_dir=pdfs_dir,
        pdf_sha256=pdf_sha256,
    )

    assert pdfs_dir.exists()
    assert managed_path.parent == pdfs_dir.resolve()
    assert managed_path.exists()
    assert managed_path.read_bytes() == source_pdf.read_bytes()


def test_copy_pdf_to_managed_store_is_idempotent_for_same_contents(tmp_path):
    """Verify that recopying the same PDF returns the existing managed file."""

    source_pdf = write_fake_pdf(tmp_path / "paper.pdf", b"same bytes\n")
    pdf_sha256 = sha256_file(source_pdf)
    pdfs_dir = tmp_path / "pdfs"

    first_path = copy_pdf_to_managed_store(source_pdf, pdfs_dir, pdf_sha256)
    second_path = copy_pdf_to_managed_store(source_pdf, pdfs_dir, pdf_sha256)

    assert second_path == first_path
    assert list(pdfs_dir.glob("*.pdf")) == [first_path]


def test_copy_pdf_to_managed_store_rejects_existing_wrong_contents(tmp_path):
    """Verify that a same-name managed destination with wrong bytes raises."""

    source_pdf = write_fake_pdf(tmp_path / "paper.pdf", b"correct bytes\n")
    pdf_sha256 = sha256_file(source_pdf)
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()

    destination = pdfs_dir / managed_pdf_filename(source_pdf, pdf_sha256)
    write_fake_pdf(destination, b"wrong bytes\n")

    with pytest.raises(ValueError, match="filename collision|corrupted"):
        copy_pdf_to_managed_store(source_pdf, pdfs_dir, pdf_sha256)


def test_ingest_pdf_with_metadata_defaults_cleanly_to_managed_storage(
    tmp_path,
    monkeypatch,
):
    """Verify that managed ingest stores new documents under pdfs/."""

    patch_chunking(monkeypatch)
    source_pdf = write_fake_pdf(tmp_path / "Readable Name.pdf", b"new managed doc\n")
    managed_pdf_dir = tmp_path / "state" / "pdfs"
    store = FakeStore()
    embedder = FakeEmbedder()

    outcome = cli.ingest_pdf_with_metadata(
        pdf_path=source_pdf,
        store=store,
        embedder=embedder,
        metadata=ExtractedMetadata(title="Managed Paper"),
        metadata_was_reviewed=False,
        reviewed_headings=[],
        use_llm_sectioning=False,
        storage_mode="managed",
        managed_pdf_dir=managed_pdf_dir,
    )

    document = store.inserted_documents[0]

    assert outcome.doc_id == document.doc_id
    assert outcome.already_existed is False
    assert document.storage_mode == "managed"
    assert document.pdf_path.parent == managed_pdf_dir.resolve()
    assert document.pdf_path.exists()
    assert document.pdf_path.read_bytes() == source_pdf.read_bytes()
    assert embedder.indexed_doc_ids == [document.doc_id]


def test_ingest_pdf_with_metadata_in_place_storage_keeps_original_path(
    tmp_path,
    monkeypatch,
):
    """Verify that external/in-place ingest records the source PDF path."""

    patch_chunking(monkeypatch)
    source_pdf = write_fake_pdf(tmp_path / "external.pdf", b"external bytes\n")
    managed_pdf_dir = tmp_path / "state" / "pdfs"
    store = FakeStore()
    embedder = FakeEmbedder()

    cli.ingest_pdf_with_metadata(
        pdf_path=source_pdf,
        store=store,
        embedder=embedder,
        metadata=ExtractedMetadata(title="External Paper"),
        metadata_was_reviewed=False,
        reviewed_headings=[],
        use_llm_sectioning=False,
        storage_mode="external",
        managed_pdf_dir=None,
    )

    document = store.inserted_documents[0]

    assert document.storage_mode == "external"
    assert document.pdf_path == source_pdf.resolve()
    assert not managed_pdf_dir.exists()


def test_duplicate_managed_ingest_reuses_existing_document_without_second_copy(
    tmp_path,
    monkeypatch,
):
    """Verify that same-content PDFs dedupe by hash in managed storage."""

    patch_chunking(monkeypatch)
    first_pdf = write_fake_pdf(tmp_path / "first-name.pdf", b"same document\n")
    second_pdf = write_fake_pdf(tmp_path / "better-name.pdf", b"same document\n")
    managed_pdf_dir = tmp_path / "state" / "pdfs"
    store = FakeStore()
    embedder = FakeEmbedder()

    first_outcome = cli.ingest_pdf_with_metadata(
        pdf_path=first_pdf,
        store=store,
        embedder=embedder,
        metadata=ExtractedMetadata(title="First Name"),
        metadata_was_reviewed=False,
        reviewed_headings=[],
        use_llm_sectioning=False,
        storage_mode="managed",
        managed_pdf_dir=managed_pdf_dir,
    )
    second_outcome = cli.ingest_pdf_with_metadata(
        pdf_path=second_pdf,
        store=store,
        embedder=embedder,
        metadata=ExtractedMetadata(title="Better Name"),
        metadata_was_reviewed=False,
        reviewed_headings=[],
        use_llm_sectioning=False,
        storage_mode="managed",
        managed_pdf_dir=managed_pdf_dir,
    )

    assert first_outcome.doc_id == second_outcome.doc_id
    assert first_outcome.already_existed is False
    assert second_outcome.already_existed is True
    assert len(store.inserted_documents) == 1
    assert len(list(managed_pdf_dir.glob("*.pdf"))) == 1


def test_existing_document_status_reports_missing_current_chunks(tmp_path):
    """Verify detection of existing docs without current-version chunks."""

    source_pdf = write_fake_pdf(tmp_path / "paper.pdf", b"existing doc\n")
    pdf_sha256 = sha256_file(source_pdf)
    store = FakeStore()

    document = Document.for_pdf(
        pdf_path=source_pdf,
        pdf_sha256=pdf_sha256,
        storage_mode="external",
        metadata=ExtractedMetadata(title="Existing Paper"),
    )
    store.insert_document(document)

    status = cli.existing_document_status(source_pdf, store)

    assert status is not None
    assert status.document.doc_id == document.doc_id
    assert status.has_current_pipeline is False


def test_missing_current_chunks_message_mentions_stale_pipeline(
    tmp_path,
    capsys,
):
    """Verify the user-facing wording for stale derived ingest artifacts."""

    source_pdf = write_fake_pdf(tmp_path / "paper.pdf", b"message case\n")
    document = Document.for_pdf(
        pdf_path=source_pdf,
        pdf_sha256=sha256_file(source_pdf),
        storage_mode="external",
        metadata=ExtractedMetadata(title="Existing Paper"),
    )

    cli.print_existing_document_needs_current_chunks_message(source_pdf, document)

    output = " ".join(capsys.readouterr().out.split())

    assert (
        "has not been processed with the current "
        "extraction/sectioning/chunking pipeline" in output
    )
