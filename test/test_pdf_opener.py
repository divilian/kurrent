from pathlib import Path

import kurrent.pdf_opener as pdf_opener


def test_open_pdf_uses_okular_with_page_on_linux(monkeypatch, tmp_path):
    """Linux page navigation should prefer Okular when it is available."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    popen_calls = []

    monkeypatch.setattr(pdf_opener.sys, "platform", "linux")
    monkeypatch.setattr(pdf_opener.shutil, "which", lambda name: "/usr/bin/okular")
    monkeypatch.setattr(
        pdf_opener.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    result = pdf_opener.open_pdf(pdf_path, page=5)

    assert result.success is True
    assert result.page == 5
    assert result.page_supported is True
    assert result.command == ("okular", "--page", "5", str(pdf_path))
    assert popen_calls[0][0][0] == ["okular", "--page", "5", str(pdf_path)]
    assert popen_calls[0][1]["start_new_session"] is True


def test_open_pdf_falls_back_to_xdg_open_when_okular_is_unavailable(monkeypatch, tmp_path):
    """Linux without Okular should still open the PDF, but without page support."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    popen_calls = []

    monkeypatch.setattr(pdf_opener.sys, "platform", "linux")
    monkeypatch.setattr(pdf_opener.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        pdf_opener.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    result = pdf_opener.open_pdf(pdf_path, page=5)

    assert result.success is True
    assert result.page == 5
    assert result.page_supported is False
    assert result.command == ("xdg-open", str(pdf_path))
    assert popen_calls[0][0][0] == ["xdg-open", str(pdf_path)]


def test_open_pdf_reports_missing_file_without_raising():
    """Missing PDFs should return a failed result instead of raising."""

    missing_path = Path("/tmp/kurrent-missing-paper.pdf")

    result = pdf_opener.open_pdf(missing_path, page=3)

    assert result.success is False
    assert result.page == 3
    assert "PDF path does not exist" in result.message


def test_open_pdf_uses_macos_open(monkeypatch, tmp_path):
    """macOS should use the system open command."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    popen_calls = []

    monkeypatch.setattr(pdf_opener.sys, "platform", "darwin")
    monkeypatch.setattr(
        pdf_opener.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    result = pdf_opener.open_pdf(pdf_path, page=2)

    assert result.success is True
    assert result.page_supported is False
    assert result.command == ("open", str(pdf_path))
    assert popen_calls[0][0][0] == ["open", str(pdf_path)]
