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
    assert "1. Nowak and May 1992: 1a pp. 3–4" in captured.out




def test_print_converse_sources_lists_passage_shortcuts_in_retrieval_order(capsys):
    """Grouped sources should expose compact shortcuts like 1a and 1b."""

    pdf_path = Path("/tmp/paper.pdf")
    packets = (
        EvidencePacket(
            evidence_id=1,
            chunk_id="doc-1:v:0",
            source_label="Nowak and May 1992",
            citation="Nowak and May 1992, p. 7",
            title="A paper",
            source_name="paper.pdf",
            pdf_path=pdf_path,
            page_start=7,
            page_end=7,
            pages="p. 7",
            section=None,
            distance=0.1,
            text="Third-page excerpt.",
        ),
        EvidencePacket(
            evidence_id=2,
            chunk_id="doc-1:v:1",
            source_label="Nowak and May 1992",
            citation="Nowak and May 1992, p. 3",
            title="A paper",
            source_name="paper.pdf",
            pdf_path=pdf_path,
            page_start=3,
            page_end=3,
            pages="p. 3",
            section=None,
            distance=0.2,
            text="Earlier-page excerpt.",
        ),
    )
    turn = ConverseTurn(
        user_text="question",
        retrieval_query="question",
        assistant_text="answer",
        evidence=packets,
    )

    cli.print_converse_sources(turn)

    captured = capsys.readouterr()
    assert "1. Nowak and May 1992: 1a p. 7; 1b p. 3" in captured.out


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




def test_open_converse_source_can_open_lettered_passage(monkeypatch, tmp_path):
    """Source selections like 1b should open the matching grouped passage."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    packets = (
        EvidencePacket(
            evidence_id=1,
            chunk_id="doc-1:v:0",
            source_label="Nowak and May 1992",
            citation="Nowak and May 1992, p. 7",
            title="A paper",
            source_name="paper.pdf",
            pdf_path=pdf_path,
            page_start=7,
            page_end=7,
            pages="p. 7",
            section=None,
            distance=0.1,
            text="Page seven excerpt.",
        ),
        EvidencePacket(
            evidence_id=2,
            chunk_id="doc-1:v:1",
            source_label="Nowak and May 1992",
            citation="Nowak and May 1992, p. 3",
            title="A paper",
            source_name="paper.pdf",
            pdf_path=pdf_path,
            page_start=3,
            page_end=3,
            pages="p. 3",
            section=None,
            distance=0.2,
            text="Page three excerpt.",
        ),
    )
    turn = ConverseTurn(
        user_text="question",
        retrieval_query="question",
        assistant_text="answer",
        evidence=packets,
    )
    opened = []
    highlight_calls = []

    class FakeHighlightResult:
        success = False
        highlighted_pdf_path = None
        message = None

    class FakeOpenResult:
        success = True
        path = pdf_path
        page = 3
        page_supported = True
        message = None

    def fake_highlight(**kwargs):
        highlight_calls.append(kwargs)
        return FakeHighlightResult()

    monkeypatch.setattr(cli, "create_highlighted_pdf_for_research_interest", fake_highlight)
    monkeypatch.setattr(
        cli,
        "open_pdf",
        lambda path, page=None: opened.append((path, page)) or FakeOpenResult(),
    )

    cli.open_converse_source(turn, "1b")

    assert opened == [(pdf_path, 3)]
    assert highlight_calls[0]["page_start"] == 3
    assert highlight_calls[0]["fallback_excerpt"] == "Page three excerpt."


def test_open_converse_source_rejects_bad_source_number(capsys):
    """Bad /open arguments should produce a friendly message."""

    cli.open_converse_source(make_turn(), "bogus")

    captured = capsys.readouterr()
    assert "source number like 1" in captured.out


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


def test_open_converse_source_prefers_highlighted_pdf_when_available(monkeypatch, tmp_path):
    """Opening a converse source should use a temporary highlighted PDF when created."""

    pdf_path = tmp_path / "paper.pdf"
    highlighted_path = tmp_path / "paper-highlighted.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    highlighted_path.write_bytes(b"%PDF-1.4\n")
    turn = make_turn(pdf_path=pdf_path)
    opened = []
    highlight_calls = []

    class FakeHighlightResult:
        success = True
        highlighted_pdf_path = highlighted_path
        message = None

    class FakeOpenResult:
        success = True
        path = highlighted_path
        page = 3
        page_supported = True
        message = None

    def fake_highlight(**kwargs):
        highlight_calls.append(kwargs)
        return FakeHighlightResult()

    monkeypatch.setattr(cli, "create_highlighted_pdf_for_research_interest", fake_highlight)
    monkeypatch.setattr(
        cli,
        "open_pdf",
        lambda path, page=None: opened.append((path, page)) or FakeOpenResult(),
    )

    cli.open_converse_source(turn, "1", ollama_model="model-x", ollama_url="http://ollama")

    assert opened == [(highlighted_path, 3)]
    assert highlight_calls[0]["pdf_path"] == pdf_path
    assert highlight_calls[0]["page_start"] == 3
    assert highlight_calls[0]["research_interest"] == "question"
    assert highlight_calls[0]["model"] == "model-x"
    assert highlight_calls[0]["ollama_url"] == "http://ollama"
    assert highlight_calls[0]["fallback_excerpt"] == "Relevant excerpt."
