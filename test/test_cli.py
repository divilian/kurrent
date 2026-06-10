from pathlib import Path

import pytest

from kurrent import cli
from kurrent import sectioner


def test_ingest_defaults_to_crossref_metadata():
    """Verify that Crossref metadata is the default ingest metadata mode."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "paper.pdf"])

    assert args.command == "ingest"
    assert args.path == Path("paper.pdf")
    assert args.metadata_mode == "crossref"
    assert args.recursive is False
    assert args.assume_yes is False


def test_ingest_accepts_local_metadata_flag():
    """Verify that --local-metadata selects local-only metadata extraction."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--local-metadata", "paper.pdf"])

    assert args.metadata_mode == "local"


def test_ingest_accepts_crossref_metadata_flag():
    """Verify that --crossref-metadata explicitly selects Crossref metadata."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--crossref-metadata", "paper.pdf"])

    assert args.metadata_mode == "crossref"


def test_metadata_flags_are_mutually_exclusive():
    """Verify that local and Crossref metadata modes cannot both be selected."""

    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ingest",
                "--local-metadata",
                "--crossref-metadata",
                "paper.pdf",
            ]
        )


def test_recursive_and_yes_flags_parse_together():
    """Verify that -y and -r can be combined for noninteractive batch ingest."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "-y", "-r", "pdfs"])

    assert args.path == Path("pdfs")
    assert args.recursive is True
    assert args.assume_yes is True


def test_state_dir_global_option_parses_before_subcommand():
    """Verify that --state-dir is accepted as a top-level kurrent option."""

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--state-dir",
            "/tmp/kurrent-test-state",
            "ingest",
            "paper.pdf",
        ]
    )

    assert args.state_dir == Path("/tmp/kurrent-test-state")
    assert args.path == Path("paper.pdf")


def test_ingest_targets_accepts_single_pdf(tmp_path):
    """Verify that one valid PDF file is selected for non-recursive ingest."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF- fake test pdf")

    targets = cli.ingest_targets(pdf_path, recursive=False)

    assert targets == [pdf_path.resolve()]


def test_ingest_targets_rejects_directory_without_recursive(tmp_path):
    """Verify that directory ingest requires -r/--recursive."""

    with pytest.raises(cli.CliUsageError) as excinfo:
        cli.ingest_targets(tmp_path, recursive=False)

    assert "Directory ingest requires -r/--recursive" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_recursive_ingest_rejects_file(tmp_path):
    """Verify that recursive ingest requires a directory, not a file."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(cli.CliUsageError) as excinfo:
        cli.ingest_targets(pdf_path, recursive=True)

    assert "Recursive ingest requires a directory" in str(excinfo.value)
    assert "Got a file instead" in str(excinfo.value)


def test_ingest_targets_recursively_finds_pdfs(tmp_path):
    """Verify that recursive ingest discovers PDFs below a directory."""

    first_pdf = tmp_path / "a.pdf"
    second_pdf = tmp_path / "nested" / "b.pdf"
    non_pdf = tmp_path / "notes.txt"

    second_pdf.parent.mkdir()
    first_pdf.write_bytes(b"%PDF- first")
    second_pdf.write_bytes(b"%PDF- second")
    non_pdf.write_text("not a PDF", encoding="utf-8")

    targets = cli.ingest_targets(tmp_path, recursive=True)

    assert targets == sorted([first_pdf.resolve(), second_pdf.resolve()])


def test_sectioner_accepts_common_and_numbered_headings():
    """Verify that plausible section headings are accepted."""

    assert sectioner._looks_like_heading("Abstract")
    assert sectioner._looks_like_heading("I. Introduction")
    assert sectioner._looks_like_heading("II. The Model")
    assert sectioner._looks_like_heading("2.3 Simulation Results")


def test_sectioner_rejects_front_matter_junk():
    """Verify that author, affiliation, and manuscript-junk lines are rejected."""

    assert not sectioner._looks_like_heading(
        "Feng Fu1,2, Christoph Hauert1,3, Martin A. Nowak1,4,*, "
        "and Long Wang2,†"
    )
    assert not sectioner._looks_like_heading(
        "1Program for Evolutionary Dynamics, Harvard University, "
        "One Brattle Square, Cambridge, MA 02138, USA"
    )
    assert not sectioner._looks_like_heading("NIH Public Access")
    assert not sectioner._looks_like_heading("Author Manuscript")
    assert not sectioner._looks_like_heading("PHYSICAL REVIEW E 89, 042142 (2014)")


def test_parse_number_list_parses_comma_separated_heading_numbers():
    """Verify that comma-separated 1-based numbers are parsed as a set."""

    assert cli.parse_number_list("1, 3, 5", maximum=5) == {1, 3, 5}


def test_parse_number_list_rejects_out_of_range_numbers():
    """Verify that heading removal numbers must be within range."""

    with pytest.raises(ValueError, match="out of range"):
        cli.parse_number_list("1, 6", maximum=5)


def test_sectioner_dedupe_preserving_order_is_case_insensitive():
    """Verify that duplicate headings are removed without reordering."""

    values = [
        "Abstract",
        "Introduction",
        "abstract",
        "Methods",
        "INTRODUCTION",
    ]

    assert sectioner._dedupe_preserving_order(values) == [
        "Abstract",
        "Introduction",
        "Methods",
    ]



def test_refresh_metadata_command_defaults_to_auto_method():
    """Verify that refresh-metadata can inspect all documents by default."""

    parser = cli.build_parser()
    args = parser.parse_args(["refresh-metadata", "--dry-run"])

    assert args.command == "refresh-metadata"
    assert args.query == []
    assert args.method == "auto"
    assert args.dry_run is True
    assert args.assume_yes is False


def test_refresh_metadata_command_accepts_query_and_yes_flag():
    """Verify that refresh-metadata can target matching docs and apply updates."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "refresh-metadata",
        "--method",
        "llm",
        "-y",
        "design08",
    ])

    assert args.query == ["design08"]
    assert args.method == "llm"
    assert args.assume_yes is True


def test_prompt_apply_metadata_refresh_accepts_yes_no_all_and_quit(monkeypatch):
    """Verify metadata refresh prompts support per-update and batch choices."""

    for typed, expected in [
        ("", "no"),
        ("n", "no"),
        ("y", "yes"),
        ("a", "all"),
        ("q", "quit"),
    ]:
        monkeypatch.setattr("builtins.input", lambda _prompt, typed=typed: typed)
        assert cli.prompt_apply_metadata_refresh() == expected


def test_prompt_apply_metadata_refresh_reprompts_on_invalid_answer(monkeypatch, capsys):
    """Verify invalid metadata refresh prompt responses are rejected cleanly."""

    answers = iter(["wat", "a"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert cli.prompt_apply_metadata_refresh() == "all"
    assert "Please enter y, n, a, or q." in capsys.readouterr().out


def test_metadata_document_summary_prints_pdf_path_and_missing_marker(capsys, tmp_path):
    """Metadata search summaries should show where the PDF lives."""

    from kurrent.schema import DocumentHit

    missing_pdf = tmp_path / "missing.pdf"
    hit = DocumentHit(
        doc_id="doc-1",
        path=missing_pdf,
        title="Birds of a Feather",
        authors="Miller McPherson",
        year=2001,
    )

    cli.print_document_summary(hit, index=1, total=1)

    output = capsys.readouterr().out
    assert f"pdf: {missing_pdf}" in output
    assert "[MISSING]" in output


def test_metadata_document_detail_prints_pdf_path_without_missing_marker(capsys, tmp_path):
    """Metadata search details should include the PDF path too."""

    from kurrent.schema import DocumentHit

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    hit = DocumentHit(
        doc_id="doc-1",
        path=pdf_path,
        title="Birds of a Feather",
        authors="Miller McPherson",
        year=2001,
    )

    cli.print_document_detail(hit, index=1, total=1)

    output = capsys.readouterr().out
    assert f"pdf: {pdf_path}" in output
    assert "[MISSING]" not in output



def test_metadata_document_detail_prints_management_fields(capsys, tmp_path):
    """The metadata details view should expose document-management fields."""

    from datetime import datetime, timezone

    from kurrent.schema import DocumentHit
    from test.factories import make_document

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    ingested_at = datetime(2026, 6, 9, 12, 30, tzinfo=timezone.utc)
    document = make_document(
        doc_id="doc-1",
        pdf_sha256="abc123",
        storage_mode="external",
        pdf_path=pdf_path,
        ingested_at=ingested_at,
        title="Birds of a Feather",
        authors="Miller McPherson",
        year=2001,
        doi="10.1146/annurev.soc.27.1.415",
    )

    class FakeStore:
        def get_document(self, doc_id):
            assert doc_id == "doc-1"
            return document

        def get_document_pipeline_state(self, doc_id):
            assert doc_id == "doc-1"
            return {
                "pipeline_fingerprint": "pipeline-v1",
                "status": "ok",
                "message": None,
                "updated_at": "2026-06-09T12:31:00+00:00",
            }

    hit = DocumentHit(
        doc_id="doc-1",
        path=pdf_path,
        title="Birds of a Feather",
        authors="Miller McPherson",
        year=2001,
        score=0.875,
        best_chunk_id="doc-1:chunker:3",
    )

    cli.print_document_detail(
        hit,
        index=1,
        total=1,
        state_store=FakeStore(),
    )

    output = capsys.readouterr().out
    assert "doc_id: doc-1" in output
    assert "doi: 10.1146/annurev.soc.27.1.415" in output
    assert "score: 0.8750" in output
    assert "best chunk: doc-1:chunker:3" in output
    assert "storage: external" in output
    assert "ingested: 2026-06-09 12:30:00+00:00" in output
    assert "pdf sha256: abc123" in output
    assert "pipeline status: ok" in output
    assert "pipeline updated: 2026-06-09T12:31:00+00:00" in output
    assert "pipeline fingerprint: pipeline-v1" in output

def test_full_text_chunk_summary_can_print_parent_pdf_path(capsys, tmp_path):
    """Full-text chunk results should optionally show the parent PDF path."""

    from kurrent.schema import ChunkHit
    from test.factories import make_document

    pdf_path = tmp_path / "article.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = make_document(
        doc_id="doc-1",
        pdf_path=pdf_path,
        title="Example Article",
        authors="Jane Author",
        year=2024,
    )

    class FakeStore:
        def get_document(self, doc_id):
            assert doc_id == "doc-1"
            return document

    hit = ChunkHit(
        chunk_id="doc-1:chunker:0",
        distance=None,
        text="homophily appears in this chunk",
        path=pdf_path,
        title="Example Article",
    )

    cli.print_chunk_summary(
        hit,
        index=1,
        total=1,
        search_text="homophily",
        state_store=FakeStore(),
        show_pdf_path=True,
    )

    output = capsys.readouterr().out
    assert f"pdf: {pdf_path}" in output
    assert "[MISSING]" not in output


def test_document_result_prompt_includes_open_only_when_pdf_exists(monkeypatch, tmp_path):
    """Document result prompts should offer open only for existing PDFs."""

    from kurrent.schema import DocumentHit

    existing_pdf = tmp_path / "exists.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4\n")
    missing_pdf = tmp_path / "missing.pdf"
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)

    existing_hit = DocumentHit(
        doc_id="doc-1",
        path=existing_pdf,
        title="Existing",
        authors=None,
        year=None,
    )
    missing_hit = DocumentHit(
        doc_id="doc-2",
        path=missing_pdf,
        title="Missing",
        authors=None,
        year=None,
    )

    assert cli.prompt_document_result_action(existing_hit) == ""
    assert cli.prompt_document_result_action(missing_hit) == ""

    assert "[o]pen PDF" in prompts[0]
    assert "[o]pen PDF" not in prompts[1]
    assert "[d]etails" in prompts[0]
    assert "[e]dit metadata" in prompts[0]
    assert "[q]uit" in prompts[0]


def test_present_document_hits_open_choice_opens_pdf(monkeypatch, tmp_path, capsys):
    """The document-result menu should open PDFs without editing metadata."""

    from types import SimpleNamespace

    from kurrent.schema import DocumentHit

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    hit = DocumentHit(
        doc_id="doc-1",
        path=pdf_path,
        title="Birds of a Feather",
        authors="Miller McPherson",
        year=2001,
    )
    opened = []
    answers = iter(["o", "q"])

    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    def fake_open_pdf(path, page=None):
        opened.append((Path(path), page))
        return SimpleNamespace(
            success=True,
            message=None,
            path=Path(path),
            page=page,
            page_supported=False,
        )

    monkeypatch.setattr(cli, "open_pdf", fake_open_pdf)

    cli.present_document_hits([hit])

    assert opened == [(pdf_path, None)]
    output = capsys.readouterr().out
    assert f"Opened PDF: {pdf_path}" in output


def test_present_document_hits_rejects_open_when_pdf_missing(monkeypatch, tmp_path, capsys):
    """The open command should not run for a missing PDF path."""

    from kurrent.schema import DocumentHit

    missing_pdf = tmp_path / "missing.pdf"
    hit = DocumentHit(
        doc_id="doc-1",
        path=missing_pdf,
        title="Birds of a Feather",
        authors="Miller McPherson",
        year=2001,
    )
    answers = iter(["o", "q"])

    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        cli,
        "open_pdf",
        lambda *args, **kwargs: pytest.fail("open_pdf should not be called"),
    )

    cli.present_document_hits([hit])

    output = capsys.readouterr().out
    assert "Please press Enter, or type d, e, or q." in output


def test_converse_accepts_initial_research_question_arguments():
    """Verify converse can take the first research question from argv."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "converse",
        "agents",
        "rewiring",
        "local",
        "networks",
    ])

    assert args.command == "converse"
    assert args.research_question == [
        "agents",
        "rewiring",
        "local",
        "networks",
    ]


def test_converse_initial_research_question_can_follow_options():
    """Verify converse options still parse before an initial research question."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "converse",
        "--limit",
        "4",
        "network",
        "rewiring",
    ])

    assert args.limit == 4
    assert args.research_question == ["network", "rewiring"]


def test_converse_debug_options_parse_before_initial_question():
    """Verify converse accepts semantic retrieval debug flags."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "converse",
        "--debug",
        "--debug-candidates",
        "12",
        "--debug-grep",
        "memex|PKB",
        "personal",
        "knowledge",
        "base",
    ])

    assert args.command == "converse"
    assert args.debug is True
    assert args.debug_candidates == 12
    assert args.debug_grep == ["memex|PKB"]
    assert args.research_question == ["personal", "knowledge", "base"]


def test_search_debug_options_parse():
    """Verify semantic search accepts retrieval debug flags."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "search",
        "--debug",
        "--debug-candidates",
        "25",
        "--debug-grep",
        "Vannevar",
        "personal",
        "knowledge",
        "base",
    ])

    assert args.command == "search"
    assert args.debug is True
    assert args.debug_candidates == 25
    assert args.debug_grep == ["Vannevar"]
    assert args.query == ["personal", "knowledge", "base"]


def test_print_semantic_debug_report_prints_semantic_and_grep_sections(capsys, tmp_path):
    """Verify retrieval debug output includes semantic, lexical, and grep info."""

    from kurrent.searcher import Searcher
    from kurrent.schema import VectorChunkMatch
    from test.factories import make_chunk, make_document

    document = make_document(
        doc_id="doc-debug",
        pdf_path=tmp_path / "memex.pdf",
        title="Still Building the Memex",
        authors="Stephen Davies",
        year=2011,
    )
    chunk = make_chunk(
        document.doc_id,
        0,
        "A personal knowledge base stores a person's memories for later query.",
        page_start=1,
        page_end=2,
    )
    from kurrent.state_store import StateStore

    store = StateStore(tmp_path / "kurrent.db")
    store.insert_document(document)
    store.insert_chunks([chunk])

    class FakeEmbedder:
        model_name = "fake-model"
        collection_name = "fake-collection"

        def query_chunks(self, search_text, n_results=10, max_distance=None, exclude_doc_ids=None):
            return [VectorChunkMatch(chunk_id=chunk.chunk_id, distance=0.1234)]

    searcher = Searcher(state_store=store, embedder=FakeEmbedder())

    cli.print_semantic_debug_report(
        searcher,
        "personal knowledge base",
        n_results=5,
        grep_patterns=["memex|PKB"],
    )

    output = capsys.readouterr().out
    store.close()
    assert "Semantic debug report" in output
    assert "Embedding model: fake-model" in output
    assert "Semantic chunks returned: 1" in output
    assert "Exact chunk-text search hits for full query: 1" in output
    assert "Metadata search hits for full query" in output
    assert "Grep diagnostics for /memex|PKB/i" in output
    assert "Still Building the Memex" in output
