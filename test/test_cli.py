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

    with pytest.raises(IsADirectoryError):
        cli.ingest_targets(tmp_path, recursive=False)


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

    assert sectioner.looks_like_heading("Abstract")
    assert sectioner.looks_like_heading("I. Introduction")
    assert sectioner.looks_like_heading("II. The Model")
    assert sectioner.looks_like_heading("2.3 Simulation Results")


def test_sectioner_rejects_front_matter_junk():
    """Verify that author, affiliation, and manuscript-junk lines are rejected."""

    assert not sectioner.looks_like_heading(
        "Feng Fu1,2, Christoph Hauert1,3, Martin A. Nowak1,4,*, "
        "and Long Wang2,†"
    )
    assert not sectioner.looks_like_heading(
        "1Program for Evolutionary Dynamics, Harvard University, "
        "One Brattle Square, Cambridge, MA 02138, USA"
    )
    assert not sectioner.looks_like_heading("NIH Public Access")
    assert not sectioner.looks_like_heading("Author Manuscript")
    assert not sectioner.looks_like_heading("PHYSICAL REVIEW E 89, 042142 (2014)")


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

    assert sectioner.dedupe_preserving_order(values) == [
        "Abstract",
        "Introduction",
        "Methods",
    ]
