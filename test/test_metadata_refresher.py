from pathlib import Path
from types import SimpleNamespace

from kurrent.metadata_refresher import (
    MetadataRefreshProposal,
    assess_document_metadata,
    metadata_updates_for_document,
    _metadata_from_ollama_payload,
)
from kurrent.schema import ExtractedMetadata
from test.factories import make_document


def test_assess_document_metadata_flags_obviously_bad_zotero_like_metadata():
    document = make_document(
        pdf_path=Path("/tmp/ZOTERO123/fulltext.pdf"),
        title="plcb-02-10-11 1284..1291",
        authors="design08",
        year=2020,
        doi=None,
    )

    assessment = assess_document_metadata(document)

    assert assessment.needs_refresh
    assert any("authors" in reason for reason in assessment.reasons)


def test_assess_document_metadata_accepts_good_core_metadata():
    document = make_document(
        title="Evolutionary Dynamics on Graphs",
        authors="Nowak and May",
        year=1992,
        doi="10.123/example",
    )

    assessment = assess_document_metadata(document)

    assert not assessment.needs_refresh
    assert assessment.reasons == ()


def test_metadata_updates_replace_only_bad_fields_by_default():
    document = make_document(
        title="Good Existing Title",
        authors="design08",
        year=2020,
        doi=None,
    )
    proposal = MetadataRefreshProposal(
        metadata=ExtractedMetadata(
            title="Better Crossref Title",
            authors="Nowak and May",
            year=1992,
            doi="10.123/example",
        ),
        source="llm",
        confidence="medium",
        reason="test",
    )

    updates = metadata_updates_for_document(document, proposal)

    assert updates == {
        "authors": "Nowak and May",
        "doi": "10.123/example",
    }


def test_metadata_updates_can_replace_all_crossref_fields():
    document = make_document(
        title="Good Existing Title",
        authors="design08",
        year=2020,
        doi=None,
    )
    proposal = MetadataRefreshProposal(
        metadata=ExtractedMetadata(
            title="Better Crossref Title",
            authors="Nowak and May",
            year=1992,
            doi="10.123/example",
        ),
        source="crossref",
        confidence="high",
        reason="test",
    )

    updates = metadata_updates_for_document(document, proposal, replace_all=True)

    assert updates == {
        "title": "Better Crossref Title",
        "authors": "Nowak and May",
        "year": 1992,
        "doi": "10.123/example",
    }


def test_metadata_from_ollama_payload_sanitizes_bad_values():
    metadata, confidence, reason = _metadata_from_ollama_payload(
        {
            "title": "Microsoft Word - manuscript.docx",
            "authors": "design08",
            "year": "2006",
            "doi": "10.1234/example",
            "confidence": "high",
            "reason": "visible first page metadata",
        }
    )

    assert metadata == ExtractedMetadata(
        title=None,
        authors=None,
        year=2006,
        doi="10.1234/example",
    )
    assert confidence == "high"
    assert reason == "visible first page metadata"


def test_metadata_updates_name_cases_all_caps_authors():
    document = make_document(
        title="Good Existing Title",
        authors="design08",
        year=2020,
    )
    proposal = MetadataRefreshProposal(
        metadata=ExtractedMetadata(
            authors="WILLIAM W. COHEN and YORAM SINGER",
        ),
        source="llm",
        confidence="medium",
        reason="test",
    )

    updates = metadata_updates_for_document(document, proposal)

    assert updates == {
        "authors": "William W. Cohen and Yoram Singer",
    }


def test_metadata_from_ollama_payload_name_cases_all_caps_authors():
    metadata, confidence, reason = _metadata_from_ollama_payload(
        {
            "title": "Learning to Classify Text",
            "authors": "WILLIAM W. COHEN and YORAM SINGER",
            "year": "1996",
            "doi": None,
            "confidence": "medium",
            "reason": "visible first page metadata",
        }
    )

    assert metadata.authors == "William W. Cohen and Yoram Singer"
    assert confidence == "medium"
    assert reason == "visible first page metadata"


def test_ensure_ollama_available_returns_false_when_already_reachable(monkeypatch):
    from kurrent import metadata_refresher

    monkeypatch.setattr(metadata_refresher, "_ollama_is_reachable", lambda url: True)

    assert metadata_refresher.ensure_ollama_available() is False


def test_ensure_ollama_available_starts_ollama_when_unreachable(monkeypatch):
    from kurrent import metadata_refresher

    reachable_checks = iter([False, False, True])
    started_commands = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            started_commands.append(command)

    monkeypatch.setattr(
        metadata_refresher,
        "_ollama_is_reachable",
        lambda url: next(reachable_checks),
    )
    monkeypatch.setattr(metadata_refresher.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(metadata_refresher.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(metadata_refresher.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(metadata_refresher.time, "sleep", lambda seconds: None)

    assert metadata_refresher.ensure_ollama_available(startup_timeout_seconds=1) is True
    assert started_commands == [["/usr/bin/ollama", "serve"]]


def test_ensure_ollama_available_raises_when_cli_missing(monkeypatch):
    import pytest
    from kurrent import metadata_refresher

    monkeypatch.setattr(metadata_refresher, "_ollama_is_reachable", lambda url: False)
    monkeypatch.setattr(metadata_refresher.shutil, "which", lambda name: None)

    with pytest.raises(metadata_refresher.MetadataRefreshError, match="ollama"):
        metadata_refresher.ensure_ollama_available()
