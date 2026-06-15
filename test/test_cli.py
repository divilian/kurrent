from pathlib import Path

import pytest

from kurrent import cli
from kurrent import sectioner


def test_ingest_defaults_to_crossref_metadata():
    """Verify that Crossref metadata is the default ingest metadata mode."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "paper.pdf"])

    assert args.command == "ingest"
    assert args.paths == [Path("paper.pdf")]
    assert args.metadata_mode == "crossref"
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


def test_yes_flag_parses_for_batch_ingest():
    """Verify that -y can be combined with directory ingest."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "-y", "pdfs"])

    assert args.paths == [Path("pdfs")]
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
    assert args.paths == [Path("paper.pdf")]


def test_ingest_accepts_multiple_paths():
    """Verify that ingest accepts any number of file and directory inputs."""

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "paper.pdf", "more-papers", "other.pdf"])

    assert args.paths == [Path("paper.pdf"), Path("more-papers"), Path("other.pdf")]


def test_ingest_targets_accepts_single_pdf(tmp_path):
    """Verify that one valid PDF file is selected for non-recursive ingest."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF- fake test pdf")

    targets = cli.ingest_targets(pdf_path)

    assert targets == [pdf_path.resolve()]


def test_ingest_targets_accepts_directory_recursively(tmp_path):
    """Verify that a directory path automatically selects PDFs recursively."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    targets = cli.ingest_targets(tmp_path)

    assert targets == [pdf_path.resolve()]


def test_ingest_parser_rejects_removed_recursive_flag():
    """Verify that the old -r flag is no longer part of the ingest interface."""

    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "-r", "pdfs"])


def test_ingest_targets_recursively_finds_pdfs(tmp_path):
    """Verify that recursive ingest discovers PDFs below a directory."""

    first_pdf = tmp_path / "a.pdf"
    second_pdf = tmp_path / "nested" / "b.pdf"
    non_pdf = tmp_path / "notes.txt"

    second_pdf.parent.mkdir()
    first_pdf.write_bytes(b"%PDF- first")
    second_pdf.write_bytes(b"%PDF- second")
    non_pdf.write_text("not a PDF", encoding="utf-8")

    targets = cli.ingest_targets(tmp_path)

    assert targets == sorted([first_pdf.resolve(), second_pdf.resolve()])


def test_ingest_targets_accepts_multiple_files_and_directories(tmp_path):
    """Verify that multiple file and directory inputs are combined and de-duplicated."""

    first_pdf = tmp_path / "a.pdf"
    second_pdf = tmp_path / "nested" / "b.pdf"
    duplicate_pdf = tmp_path / "nested" / "duplicate.pdf"

    second_pdf.parent.mkdir()
    first_pdf.write_bytes(b"%PDF- first")
    second_pdf.write_bytes(b"%PDF- second")
    duplicate_pdf.write_bytes(b"%PDF- duplicate")

    targets = cli.ingest_targets([first_pdf, second_pdf.parent, duplicate_pdf])

    assert targets == [
        first_pdf.resolve(),
        second_pdf.resolve(),
        duplicate_pdf.resolve(),
    ]


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

    def fake_open_pdf(path, page=None, **kwargs):
        opened.append((Path(path), page, kwargs))
        return SimpleNamespace(
            success=True,
            message=None,
            path=Path(path),
            page=page,
            page_supported=False,
        )

    monkeypatch.setattr(cli, "open_pdf", fake_open_pdf)

    cli.present_document_hits([hit])

    assert opened == [(pdf_path, None, {})]
    output = capsys.readouterr().out
    assert f"Opened PDF: {pdf_path}" in output


def test_ingest_metadata_review_opens_pdf_without_repeating_path(monkeypatch, tmp_path, capsys):
    """Ingest metadata review should open the PDF quietly before metadata prompts."""

    from types import SimpleNamespace

    from kurrent.schema import ExtractedMetadata

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    opened = []
    answers = iter(["", "", "", ""])

    def fake_input(prompt):
        print(prompt, end="")
        return next(answers)

    def fake_open_pdf(path, page=None, **kwargs):
        opened.append((Path(path), page, kwargs))
        return SimpleNamespace(
            success=True,
            message=None,
            path=Path(path),
            page=page,
            page_supported=False,
        )

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(cli, "open_pdf", fake_open_pdf)

    metadata = ExtractedMetadata(
        title="Cooperation, social networks",
        authors="Martín G. Zimmermann, Víctor M. Eguíluz",
        year=2005,
        doi="10.1103/physreve.72.056118",
    )

    cli.open_pdf_for_metadata_review(pdf_path, metadata)
    reviewed = cli.review_metadata(metadata)

    assert reviewed == metadata
    assert opened == [(pdf_path, None, {"prefer_managed_process": True})]

    output = capsys.readouterr().out
    assert "(Opening PDF with proposed metadata highlighted:" in output
    assert f"Opened PDF: {pdf_path}" not in output
    assert output.index("(Opening PDF") < output.index("Metadata")
    assert output.index("Type corrected values where needed.") < output.index("title [")


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


def test_ask_accepts_initial_research_question_arguments():
    """Verify ask can take the first research question from argv."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "ask",
        "agents",
        "rewiring",
        "local",
        "networks",
    ])

    assert args.command == "ask"
    assert args.research_question == [
        "agents",
        "rewiring",
        "local",
        "networks",
    ]


def test_ask_initial_research_question_can_follow_options():
    """Verify ask options still parse before an initial research question."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "ask",
        "--limit",
        "4",
        "network",
        "rewiring",
    ])

    assert args.limit == 4
    assert args.research_question == ["network", "rewiring"]


def test_ask_debug_options_parse_before_initial_question():
    """Verify ask accepts semantic retrieval debug flags."""

    parser = cli.build_parser()
    args = parser.parse_args([
        "ask",
        "--debug",
        "--debug-candidates",
        "12",
        "--debug-grep",
        "memex|PKB",
        "personal",
        "knowledge",
        "base",
    ])

    assert args.command == "ask"
    assert args.debug is True
    assert args.debug_candidates == 12
    assert args.debug_grep == ["memex|PKB"]
    assert args.research_question == ["personal", "knowledge", "base"]




def test_converse_subcommand_is_removed():
    """Verify the old converse command is no longer accepted."""

    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["converse", "agents"])

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


def test_ingest_startup_status_lines_are_indented_like_ask(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Verify ingest's startup narrative uses the muted one-space status style."""

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    state_dir = tmp_path / "state"

    class FakeStateStore:
        def __init__(self, path):
            self.path = path

        def close(self):
            pass

    class FakeEmbedder:
        def __init__(self, chroma_path):
            self.chroma_path = chroma_path

    def fake_ingest_one_pdf(**kwargs):
        return cli.IngestOutcome(doc_id="doc-1", already_existed=False)

    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "kurrent.state_store",
        types.SimpleNamespace(StateStore=FakeStateStore),
    )
    monkeypatch.setitem(
        sys.modules,
        "kurrent.embedder",
        types.SimpleNamespace(Embedder=FakeEmbedder),
    )
    monkeypatch.setattr(cli, "ingest_one_pdf", fake_ingest_one_pdf)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--state-dir",
            str(state_dir),
            "ingest",
            "--local-metadata",
            str(pdf_path),
        ]
    )

    assert cli.run_ingest(args) == 0

    output_lines = capsys.readouterr().out.splitlines()
    startup_lines = [
        line
        for line in output_lines
        if line.lstrip().startswith(
            (
                "Starting kurrent ingest...",
                "kurrent state directory",
                "Finding PDFs...",
                "PDFs selected:           ",
                "SQLite database",
                "Chroma directory",
                "Managed PDF directory",
                "PDF storage mode:",
                "Metadata mode:",
                "Sectioning mode:",
                "Loading kurrent state store...",
                "Loading embedding model / Chroma index...",
                "Ready. Beginning PDF ingest.",
            )
        )
    ]

    assert startup_lines
    assert all(line.startswith(" ") for line in startup_lines)

def test_stats_command_parses_top_authors():
    """Verify that stats accepts a configurable top-author count."""

    parser = cli.build_parser()
    args = parser.parse_args(["stats", "--top-authors", "3"])

    assert args.command == "stats"
    assert args.top_authors == 3
    assert not args.histogram


def test_stats_command_parses_histogram_aliases():
    """Verify stats accepts both histogram flag spellings."""

    parser = cli.build_parser()

    args = parser.parse_args(["stats", "--hist"])
    assert args.command == "stats"
    assert args.histogram

    args = parser.parse_args(["stats", "--histogram"])
    assert args.command == "stats"
    assert args.histogram


def test_unicode_bar_uses_partial_block_elements():
    """Verify histogram bars use Unicode block elements, including fractions."""

    assert cli.unicode_bar(0, 10, width=4) == ""
    assert cli.unicode_bar(5, 10, width=4) == "██"
    assert cli.unicode_bar(1, 10, width=1) == "▏"


def test_format_year_histogram_includes_empty_years_descending():
    """Verify year histograms include zero-count gaps from latest to earliest."""

    assert cli.format_year_histogram({2024: 2, 2022: 1}, width=4) == [
        "2024 (2): ████",
        "2023 (0): ",
        "2022 (1): ██",
    ]


def test_format_year_histogram_omits_future_years_and_aligns_counts():
    """Verify histograms ignore future metadata errors and align counts."""

    assert cli.format_year_histogram(
        {2099: 99, 2026: 13, 2025: 127, 2023: 8},
        width=8,
        max_year=2026,
    ) == [
        "2026 ( 13): ▉",
        "2025 (127): ████████",
        "2024 (  0): ",
        "2023 (  8): ▌",
    ]


def test_author_surname_counts_count_all_author_positions():
    """Verify top author stats count surnames from every author-list position."""

    class FakeDocument:
        def __init__(self, authors):
            self.authors = authors

    documents = [
        FakeDocument("John Davies, Beth Tanner, and Goofus Gallant"),
        FakeDocument("Beth Tanner and Jane Tanner"),
        FakeDocument("Gregor Betz"),
    ]

    assert cli.author_surname_counts(documents) == [
        ("Tanner", 3),
        ("Betz", 1),
        ("Davies", 1),
        ("Gallant", 1),
    ]


def test_stats_command_prints_database_summary(tmp_path, capsys):
    """Verify that stats prints document, chunk, and author summary counts."""

    from datetime import datetime, timezone

    from kurrent.schema import Chunk, Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    store = StateStore(state_dir / "kurrent.db")

    try:
        first_doc = Document(
            doc_id="doc-1",
            pdf_sha256="sha-1",
            storage_mode="external",
            pdf_path=tmp_path / "first.pdf",
            ingested_at=datetime.now(timezone.utc),
            title="First Paper",
            authors="John Davies, Beth Tanner, and Goofus Gallant",
            year=2020,
            doi=None,
        )
        second_doc = Document(
            doc_id="doc-2",
            pdf_sha256="sha-2",
            storage_mode="external",
            pdf_path=tmp_path / "second.pdf",
            ingested_at=datetime.now(timezone.utc),
            title="Second Paper",
            authors="Beth Tanner and Gregor Betz",
            year=2021,
            doi=None,
        )
        store.insert_document(first_doc)
        store.insert_document(second_doc)
        store.insert_chunks(
            [
                Chunk(
                    doc_id="doc-1",
                    chunker_version="test-chunker",
                    chunk_index=0,
                    text="chunk one",
                    text_sha256="chunk-sha-1",
                    page_start=1,
                    page_end=2,
                    section_index=0,
                    section_title="Introduction",
                ),
                Chunk(
                    doc_id="doc-1",
                    chunker_version="test-chunker",
                    chunk_index=1,
                    text="chunk two",
                    text_sha256="chunk-sha-2",
                    page_start=3,
                    page_end=5,
                    section_index=1,
                    section_title="Model",
                ),
                Chunk(
                    doc_id="doc-2",
                    chunker_version="test-chunker",
                    chunk_index=0,
                    text="chunk three",
                    text_sha256="chunk-sha-3",
                    page_start=1,
                    page_end=3,
                    section_index=0,
                    section_title="Introduction",
                ),
            ]
        )
    finally:
        store.close()

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--state-dir",
            str(state_dir),
            "stats",
            "--top-authors",
            "2",
            "--hist",
        ]
    )

    assert cli.run_stats(args) == 0

    output = capsys.readouterr().out

    assert f"SQLite database:        {state_dir / 'kurrent.db'}" in output
    assert "Documents:              2" in output
    assert "Chunks:                 3" in output
    assert "Document size" in output
    assert "Avg sections/document:  1.50" in output
    assert "Avg pages/document:     4.00" in output
    assert "Avg chunks/document:    1.50" in output
    assert "Tanner:  2" in output
    assert "Betz:    1" in output or "Davies:  1" in output
    assert "Documents per year" in output
    assert "2021 (1): ████████████████████████████████████████" in output
    assert "2020 (1): ████████████████████████████████████████" in output


def test_health_command_prints_database_health(tmp_path, capsys, monkeypatch):
    """Verify health reports metadata, PDF, chunk, and semantic-index checks."""

    from datetime import datetime, timezone
    import sys
    import types

    from kurrent.chunker import chunker_version
    from kurrent.pipeline import current_text_pipeline_fingerprint
    from kurrent.schema import Chunk, Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    existing_pdf = tmp_path / "existing.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    missing_pdf = tmp_path / "missing.pdf"
    store = StateStore(state_dir / "kurrent.db")

    try:
        current_doc = Document(
            doc_id="doc-current",
            pdf_sha256="sha-current",
            storage_mode="external",
            pdf_path=existing_pdf,
            ingested_at=datetime.now(timezone.utc),
            title="Current Paper",
            authors="Jane Smith",
            year=2022,
            doi="10.123/example",
        )
        stale_doc = Document(
            doc_id="doc-stale",
            pdf_sha256="sha-stale",
            storage_mode="external",
            pdf_path=existing_pdf,
            ingested_at=datetime.now(timezone.utc),
            title="Stale Paper",
            authors=None,
            year=None,
            doi=None,
        )
        missing_path_doc = Document(
            doc_id="doc-missing-path",
            pdf_sha256="sha-missing-path",
            storage_mode="external",
            pdf_path=missing_pdf,
            ingested_at=datetime.now(timezone.utc),
            title="Missing Path Paper",
            authors="John Davies",
            year=2020,
            doi=None,
        )
        store.insert_document(current_doc)
        store.insert_document(stale_doc)
        store.insert_document(missing_path_doc)
        store.insert_chunks(
            [
                Chunk(
                    doc_id="doc-current",
                    chunker_version=chunker_version(),
                    chunk_index=0,
                    text="current chunk",
                    text_sha256="chunk-current",
                ),
            ]
        )
        store.set_document_pipeline_fingerprint(
            "doc-current",
            current_text_pipeline_fingerprint(),
        )
    finally:
        store.close()

    class FakeEmbedder:
        def __init__(self, chroma_path):
            self.chroma_path = chroma_path

        def has_document(self, doc_id):
            return False

    monkeypatch.setitem(
        sys.modules,
        "kurrent.embedder",
        types.SimpleNamespace(Embedder=FakeEmbedder),
    )

    parser = cli.build_parser()
    args = parser.parse_args(["--state-dir", str(state_dir), "health"])

    assert cli.run_health(args) == 0

    output = capsys.readouterr().out

    assert "Kurrent database health" in output
    assert f"SQLite database:        {state_dir / 'kurrent.db'}" in output
    assert "Documents:              3" in output
    assert "Missing authors:        1" in output
    assert "Missing year:           1" in output
    assert "Missing DOI:            2" in output
    assert "Missing PDF paths:      1" in output
    assert "Documents with all chunks current:   1" in output
    assert "Documents with stale/missing chunks: 1" in output
    assert "Missing from index:     1 documents" in output


def test_list_command_parses_display_aliases():
    """Verify list accepts author/title display aliases."""

    parser = cli.build_parser()

    args = parser.parse_args(["list"])
    assert args.command == "list"
    assert args.list_mode == "tag"

    for flag in ["-a", "--author", "--authors"]:
        args = parser.parse_args(["list", flag])
        assert args.list_mode == "author"

    for flag in ["-t", "--title", "--titles"]:
        args = parser.parse_args(["list", flag])
        assert args.list_mode == "title"


def test_document_list_entries_format_tags_authors_and_titles(tmp_path):
    """Verify kurrent list display modes use the requested sort labels."""

    from datetime import datetime, timezone

    from kurrent.schema import Document

    now = datetime.now(timezone.utc)
    docs = [
        Document(
            doc_id="doc-1",
            pdf_sha256="sha-1",
            storage_mode="external",
            pdf_path=tmp_path / "one.pdf",
            ingested_at=now,
            title="The Surprising Effects of Convergence",
            authors="Stephen Davies and Hannah Zontine",
            year=2016,
        ),
        Document(
            doc_id="doc-2",
            pdf_sha256="sha-2",
            storage_mode="external",
            pdf_path=tmp_path / "two.pdf",
            ingested_at=now,
            title="A Smaller Paper",
            authors="Stephen Davies and Beth Tanner",
            year=2016,
        ),
    ]

    assert [label for label, _doc in cli.list_entries_for_documents(docs, "tag")] == [
        "davies2016a",
        "davies2016b",
    ]
    assert [label for label, _doc in cli.list_entries_for_documents(docs, "author")] == [
        "Davies, Stephen and Beth Tanner, 2016",
        "Davies, Stephen and Hannah Zontine, 2016",
    ]
    assert [label for label, _doc in cli.list_entries_for_documents(docs, "title")] == [
        "Smaller Paper, A (2016).",
        "Surprising Effects of Convergence, The (2016).",
    ]


def test_list_command_prints_green_document_list_and_returns_to_list(
    tmp_path,
    capsys,
    monkeypatch,
):
    """Verify list can inspect a document and return to the document menu."""

    from datetime import datetime, timezone

    from kurrent.schema import Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    store = StateStore(state_dir / "kurrent.db")

    try:
        store.insert_document(
            Document(
                doc_id="doc-1",
                pdf_sha256="sha-1",
                storage_mode="external",
                pdf_path=tmp_path / "paper.pdf",
                ingested_at=datetime.now(timezone.utc),
                title="A Mechanistic Model of Gossip",
                authors="Mari Kawakatsu, Taylor A. Kessinger, Joshua B. Plotkin",
                year=2024,
            )
        )
    finally:
        store.close()

    responses = iter(["1", "q", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    parser = cli.build_parser()
    args = parser.parse_args(["--state-dir", str(state_dir), "list"])

    assert cli.run_list(args) == 0

    output = capsys.readouterr().out
    assert "1. kawakatsu2024" in output
    assert "A Mechanistic Model of Gossip" in output
    assert "authors: Mari Kawakatsu, Taylor A. Kessinger, Joshua B. Plotkin" in output
    assert "year: 2024" in output
    assert f"pdf: {tmp_path / 'paper.pdf'}" in output


def test_list_open_pdf_returns_to_same_action_prompt_without_editing(
    tmp_path,
    capsys,
    monkeypatch,
):
    """Verify list open action does not fall through into metadata editing."""

    from datetime import datetime, timezone

    from kurrent.schema import Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    store = StateStore(state_dir / "kurrent.db")

    try:
        store.insert_document(
            Document(
                doc_id="doc-1",
                pdf_sha256="sha-1",
                storage_mode="external",
                pdf_path=pdf_path,
                ingested_at=datetime.now(timezone.utc),
                title="A Mechanistic Model of Gossip",
                authors="Mari Kawakatsu, Taylor A. Kessinger, Joshua B. Plotkin",
                year=2024,
            )
        )
    finally:
        store.close()

    opened = []

    def fake_open_document_pdf(document, *, purpose="PDF"):
        opened.append((document.doc_id, purpose))
        print(f"Opened {purpose}: {document.pdf_path}")

    def fail_edit_metadata(*_args, **_kwargs):
        raise AssertionError("open PDF action should not edit metadata")

    responses = iter(["1", "o", "q", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
    monkeypatch.setattr(cli, "open_document_pdf", fake_open_document_pdf)
    monkeypatch.setattr(cli, "edit_document_hit_metadata", fail_edit_metadata)

    parser = cli.build_parser()
    args = parser.parse_args(["--state-dir", str(state_dir), "list"])

    assert cli.run_list(args) == 0
    assert opened == [("doc-1", "PDF")]

    output = capsys.readouterr().out
    assert output.count("A Mechanistic Model of Gossip") == 1
    assert f"Opened PDF: {pdf_path}" in output


def test_title_listing_capitalizes_after_moving_leading_article():
    """Verify title list display capitalizes the exposed first word."""

    assert (
        cli.title_for_listing("A mechanistic model of gossip, reputations, and cooperation")
        == "Mechanistic model of gossip, reputations, and cooperation, A"
    )


def test_duplicate_candidates_ignore_dismissed_pairs(tmp_path):
    """Verify possible duplicate pairs omit user-dismissed non-duplicates."""

    from datetime import datetime, timezone

    from kurrent.schema import Document
    from kurrent.state_store import StateStore

    store = StateStore(tmp_path / "kurrent.db")
    try:
        doc_a = Document(
            doc_id="doc-a",
            pdf_sha256="sha-a",
            storage_mode="external",
            pdf_path=tmp_path / "a.pdf",
            ingested_at=datetime.now(timezone.utc),
            title="Same Title",
            authors="Jane Smith",
            year=2020,
            doi="https://doi.org/10.123/EXAMPLE",
        )
        doc_b = Document(
            doc_id="doc-b",
            pdf_sha256="sha-b",
            storage_mode="external",
            pdf_path=tmp_path / "b.pdf",
            ingested_at=datetime.now(timezone.utc),
            title="Same Title",
            authors="Jane Smith",
            year=2020,
            doi="10.123/example",
        )
        store.insert_document(doc_a)
        store.insert_document(doc_b)

        candidates = cli.duplicate_candidates_for_documents(store.list_documents(), store)
        assert [(candidate.doc_a.doc_id, candidate.doc_b.doc_id, candidate.reason) for candidate in candidates] == [
            ("doc-a", "doc-b", "same DOI")
        ]

        store.record_duplicate_decision("doc-a", "doc-b", reason="same DOI")
        assert cli.duplicate_candidates_for_documents(store.list_documents(), store) == []
    finally:
        store.close()


def test_health_reports_possible_duplicate_groups(tmp_path, capsys, monkeypatch):
    """Verify health includes duplicate group counts."""

    from datetime import datetime, timezone
    import sys
    import types

    from kurrent.schema import Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    pdf_path = tmp_path / "existing.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    store = StateStore(state_dir / "kurrent.db")
    try:
        store.insert_document(
            Document(
                doc_id="doc-a",
                pdf_sha256="sha-a",
                storage_mode="external",
                pdf_path=pdf_path,
                ingested_at=datetime.now(timezone.utc),
                title="Same Title",
                authors="Jane Smith",
                year=2020,
                doi="10.123/example",
            )
        )
        store.insert_document(
            Document(
                doc_id="doc-b",
                pdf_sha256="sha-b",
                storage_mode="external",
                pdf_path=pdf_path,
                ingested_at=datetime.now(timezone.utc),
                title="Same Title",
                authors="Jane Smith",
                year=2020,
                doi="10.123/example",
            )
        )
    finally:
        store.close()

    class FakeEmbedder:
        def __init__(self, chroma_path):
            self.chroma_path = chroma_path

        def has_document(self, doc_id):
            return True

    monkeypatch.setitem(
        sys.modules,
        "kurrent.embedder",
        types.SimpleNamespace(Embedder=FakeEmbedder),
    )

    parser = cli.build_parser()
    args = parser.parse_args(["--state-dir", str(state_dir), "health"])

    assert cli.run_health(args) == 0

    output = capsys.readouterr().out
    assert "Possible duplicates" in output
    assert "Same DOI groups:                   1" in output
    assert "Same title/year/author groups:     1" in output


def test_dedupe_can_mark_pair_as_not_duplicate(tmp_path, capsys, monkeypatch):
    """Verify dedupe persists a not-duplicate decision."""

    from datetime import datetime, timezone
    import sys
    import types

    from kurrent.schema import Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    store = StateStore(state_dir / "kurrent.db")
    try:
        store.insert_document(
            Document(
                doc_id="doc-a",
                pdf_sha256="sha-a",
                storage_mode="external",
                pdf_path=tmp_path / "a.pdf",
                ingested_at=datetime.now(timezone.utc),
                title="Same Title",
                authors="Jane Smith",
                year=2020,
                doi="10.123/example",
            )
        )
        store.insert_document(
            Document(
                doc_id="doc-b",
                pdf_sha256="sha-b",
                storage_mode="external",
                pdf_path=tmp_path / "b.pdf",
                ingested_at=datetime.now(timezone.utc),
                title="Same Title",
                authors="Jane Smith",
                year=2020,
                doi="10.123/example",
            )
        )
    finally:
        store.close()

    class FakeEmbedder:
        def __init__(self, chroma_path):
            self.chroma_path = chroma_path

    monkeypatch.setitem(
        sys.modules,
        "kurrent.embedder",
        types.SimpleNamespace(Embedder=FakeEmbedder),
    )
    responses = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    parser = cli.build_parser()
    args = parser.parse_args(["--state-dir", str(state_dir), "dedupe"])

    assert cli.run_dedupe(args) == 0

    store = StateStore(state_dir / "kurrent.db")
    try:
        assert store.duplicate_pair_is_ignored("doc-a", "doc-b")
    finally:
        store.close()

    output = capsys.readouterr().out
    assert "Marked this pair as not duplicates." in output


def test_dedupe_can_open_each_pdf_separately(tmp_path, capsys, monkeypatch):
    """Verify dedupe offers separate open-1/open-2 actions instead of open-both."""

    from datetime import datetime, timezone
    import sys
    import types

    from kurrent.schema import Document
    from kurrent.state_store import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4\n%%EOF\n")
    pdf_b.write_bytes(b"%PDF-1.4\n%%EOF\n")

    store = StateStore(state_dir / "kurrent.db")
    try:
        store.insert_document(
            Document(
                doc_id="doc-a",
                pdf_sha256="sha-a",
                storage_mode="external",
                pdf_path=pdf_a,
                ingested_at=datetime.now(timezone.utc),
                title="Same Title",
                authors="Jane Smith",
                year=2020,
                doi="10.123/example",
            )
        )
        store.insert_document(
            Document(
                doc_id="doc-b",
                pdf_sha256="sha-b",
                storage_mode="external",
                pdf_path=pdf_b,
                ingested_at=datetime.now(timezone.utc),
                title="Same Title",
                authors="Jane Smith",
                year=2020,
                doi="10.123/example",
            )
        )
    finally:
        store.close()

    class FakeEmbedder:
        def __init__(self, chroma_path):
            self.chroma_path = chroma_path

    opened = []

    def fake_open_document_pdf(document, *, purpose="PDF"):
        opened.append((purpose, document.doc_id))
        return True

    monkeypatch.setitem(
        sys.modules,
        "kurrent.embedder",
        types.SimpleNamespace(Embedder=FakeEmbedder),
    )
    monkeypatch.setattr(cli, "open_document_pdf", fake_open_document_pdf)
    responses = iter(["o1", "o2", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    parser = cli.build_parser()
    args = parser.parse_args(["--state-dir", str(state_dir), "dedupe"])

    assert cli.run_dedupe(args) == 0
    assert opened == [("PDF 1", "doc-a"), ("PDF 2", "doc-b")]

    _ = capsys.readouterr().out
    prompt = []
    monkeypatch.setattr("builtins.input", lambda text="": prompt.append(text) or "q")
    assert cli.prompt_dedupe_action() == "q"
    assert "[o1]open 1, [o2]open 2" in prompt[0]
    assert "open both" not in prompt[0].lower()


def test_metadata_search_quit_does_not_reference_ingest_section_prefetcher(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Verify q from metadata search exits cleanly without ingest-only cleanup."""

    import sys
    import types

    from kurrent.schema import DocumentHit

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "kurrent.db").write_text("fake sqlite marker")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    class FakeStateStore:
        def __init__(self, path):
            self.path = path
            self.closed = False

        def close(self):
            self.closed = True

    class FakeSearcher:
        def __init__(self, state_store, embedder=None):
            self.state_store = state_store
            self.embedder = embedder

        def metadata_search(self, query, limit=10):
            return [
                DocumentHit(
                    doc_id="doc-1",
                    path=pdf_path,
                    title="A Test Paper",
                    authors="Jane Smith",
                    year=2024,
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "kurrent.state_store",
        types.SimpleNamespace(StateStore=FakeStateStore),
    )
    monkeypatch.setitem(
        sys.modules,
        "kurrent.searcher",
        types.SimpleNamespace(Searcher=FakeSearcher),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "q")

    parser = cli.build_parser()
    args = parser.parse_args([
        "--state-dir",
        str(state_dir),
        "search",
        "--metadata",
        "test",
    ])

    assert cli.run_search(args) == 0

    output = capsys.readouterr().out
    assert "Metadata search: 'test'" in output
    assert "A Test Paper" in output
