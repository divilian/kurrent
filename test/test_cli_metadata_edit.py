from pathlib import Path
from types import SimpleNamespace

import kurrent.cli as cli
from kurrent.schema import DocumentHit


def make_document(
    doc_id="doc-1",
    pdf_path=Path("/tmp/paper.pdf"),
    title="plcb-02-10-11 1284..1291",
    authors="design08",
    year=2020,
    doi=None,
):
    return SimpleNamespace(
        doc_id=doc_id,
        pdf_path=pdf_path,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
    )


def make_hit(
    doc_id="doc-1",
    path=Path("/tmp/paper.pdf"),
    title="plcb-02-10-11 1284..1291",
    authors="design08",
    year=2020,
    score=None,
    best_chunk_id=None,
):
    return DocumentHit(
        doc_id=doc_id,
        path=path,
        title=title,
        authors=authors,
        year=year,
        score=score,
        best_chunk_id=best_chunk_id,
    )


class FakeStore:
    def __init__(self, document):
        self.document = document
        self.update_calls = []

    def get_document(self, doc_id):
        if doc_id != self.document.doc_id:
            return None
        return self.document

    def update_document_metadata(self, doc_id, **updates):
        self.update_calls.append((doc_id, updates))
        for key, value in updates.items():
            setattr(self.document, key, value)


def test_edit_document_hit_metadata_updates_changed_fields_and_refreshes_hit(monkeypatch):
    """Metadata editing should update SQLite and return a refreshed DocumentHit."""

    document = make_document()
    store = FakeStore(document)
    opened = []

    monkeypatch.setattr(cli, "open_pdf_for_metadata_edit", opened.append)
    monkeypatch.setattr(
        cli,
        "review_metadata",
        lambda metadata: SimpleNamespace(
            title="Network reciprocity paper",
            authors="Nowak and May",
            year=1992,
            doi="10.123/example",
        ),
    )

    refreshed_hit = cli.edit_document_hit_metadata(make_hit(), store)

    assert opened == [document]
    assert store.update_calls == [
        (
            "doc-1",
            {
                "title": "Network reciprocity paper",
                "authors": "Nowak and May",
                "year": 1992,
                "doi": "10.123/example",
            },
        )
    ]
    assert refreshed_hit.title == "Network reciprocity paper"
    assert refreshed_hit.authors == "Nowak and May"
    assert refreshed_hit.year == 1992


def test_edit_document_hit_metadata_does_not_write_when_values_are_unchanged(monkeypatch):
    """Press-Enter-to-keep metadata editing should avoid unnecessary updates."""

    document = make_document()
    store = FakeStore(document)
    opened = []

    monkeypatch.setattr(cli, "open_pdf_for_metadata_edit", opened.append)
    monkeypatch.setattr(
        cli,
        "review_metadata",
        lambda metadata: SimpleNamespace(
            title=document.title,
            authors=document.authors,
            year=document.year,
            doi=document.doi,
        ),
    )

    same_hit = make_hit()
    returned_hit = cli.edit_document_hit_metadata(same_hit, store)

    assert opened == [document]
    assert store.update_calls == []
    assert returned_hit is same_hit


def test_prompt_document_result_action_accepts_edit_choice(monkeypatch):
    """Document-result prompts should support e-for-edit metadata correction."""

    monkeypatch.setattr("builtins.input", lambda prompt: "e")

    assert cli.prompt_document_result_action() == "e"


def test_open_pdf_for_metadata_edit_uses_linux_default_viewer(monkeypatch, tmp_path):
    """PDF opening should be best-effort and use the platform default viewer."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = make_document(pdf_path=pdf_path)
    popen_calls = []

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    cli.open_pdf_for_metadata_edit(document)

    assert popen_calls
    assert popen_calls[0][0][0] == ["xdg-open", str(pdf_path)]
    assert popen_calls[0][1]["start_new_session"] is True


def test_open_pdf_for_metadata_edit_reports_missing_pdf_without_crashing(capsys):
    """Missing PDFs should not prevent the metadata edit workflow from continuing."""

    cli.open_pdf_for_metadata_edit(make_document(pdf_path=Path("/tmp/not-here.pdf")))

    captured = capsys.readouterr()
    assert "PDF path does not exist" in captured.out
