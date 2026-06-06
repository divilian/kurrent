from pathlib import Path

import kurrent.cli as cli
from kurrent.converser import ConverseTurn, EvidencePacket


def make_turn(pdf_path=Path("/tmp/paper.pdf")):
    packet = EvidencePacket(
        evidence_id=1,
        chunk_id="doc-1:section-aware-fixed-char-2000-v2:0",
        source_label="Nowak and May 1992",
        citation="Nowak and May 1992, pp. 3–4",
        title="A paper",
        source_name="paper.pdf",
        pdf_path=pdf_path,
        page_start=3,
        page_end=4,
        pages="pp. 3–4",
        section=None,
        distance=0.123,
        text="Relevant excerpt.",
    )
    return ConverseTurn(
        user_text="question",
        retrieval_query="question",
        assistant_text="answer",
        evidence=(packet,),
    )


def test_print_converse_sources_lists_latest_turn_sources(capsys):
    """The /sources command should reveal openable sources from the last answer."""

    cli.print_converse_sources(make_turn())

    captured = capsys.readouterr()
    assert "Sources from the most recent answer" in captured.out
    assert "1. Nowak and May 1992, pp. 3–4" in captured.out


def test_print_converse_sources_handles_missing_turn(capsys):
    """Before the first answer, /sources should tell the user what to do."""

    cli.print_converse_sources(None)

    captured = capsys.readouterr()
    assert "Ask a research question first" in captured.out


def test_open_converse_source_opens_pdf_to_first_page(monkeypatch, tmp_path):
    """The /open N command should open the selected source at its first page."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    turn = make_turn(pdf_path=pdf_path)
    opened = []

    class FakeOpenResult:
        success = True
        path = pdf_path
        page = 3
        page_supported = True
        message = None

    monkeypatch.setattr(
        cli,
        "open_pdf",
        lambda path, page=None: opened.append((path, page)) or FakeOpenResult(),
    )

    cli.open_converse_source(turn, "1")

    assert opened == [(pdf_path, 3)]


def test_open_converse_source_rejects_bad_source_number(capsys):
    """Bad /open arguments should produce a friendly message."""

    cli.open_converse_source(make_turn(), "bogus")

    captured = capsys.readouterr()
    assert "Usage: /open N" in captured.out


def test_handle_converse_command_reports_unknown_command(capsys):
    """Unknown slash commands should point the user to /help."""

    assert cli.handle_converse_command("/wat", make_turn()) is True

    captured = capsys.readouterr()
    assert "Unknown command: /wat" in captured.out
    assert "Type /help" in captured.out


def test_streaming_wrapped_printer_wraps_completed_words(capsys):
    """Live Ollama output should wrap without waiting for the full answer."""

    printer = cli.StreamingWrappedPrinter(width=12)
    printer.write("one two thr")
    printer.write("ee four")
    printer.finish()

    captured = capsys.readouterr()
    assert captured.out == "one two\nthree four"


def test_streaming_wrapped_printer_preserves_model_newlines(capsys):
    """Streaming wrapper should preserve explicit paragraph/list newlines."""

    printer = cli.StreamingWrappedPrinter(width=79)
    printer.write("Line one\n* item")
    printer.finish()

    captured = capsys.readouterr()
    assert captured.out == "Line one\n* item"


def test_browse_converse_sources_opens_number_and_returns_to_main(monkeypatch, tmp_path):
    """The source browser should accept bare numbers and q to return."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    turn = make_turn(pdf_path=pdf_path)
    choices = iter(["1", "q"])
    prompts = []
    opened = []

    class FakeOpenResult:
        success = True
        path = pdf_path
        page = 3
        page_supported = True
        message = None

    def fake_input(prompt):
        prompts.append(prompt)
        return next(choices)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        cli,
        "open_pdf",
        lambda path, page=None: opened.append((path, page)) or FakeOpenResult(),
    )

    cli.browse_converse_sources(turn)

    assert prompts == ["sources> ", "sources> "]
    assert opened == [(pdf_path, 3)]


def test_source_browser_q_predicate_accepts_slash_q():
    """q and /q should leave the source browser without leaving converse."""

    assert cli.is_source_browser_quit("q")
    assert cli.is_source_browser_quit("/q")
    assert not cli.is_source_browser_quit("1")
