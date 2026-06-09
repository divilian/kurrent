from pathlib import Path

import pymupdf

from kurrent.metadata_extractor import (
    clean_author_metadata_text,
    clean_metadata_text,
    clean_title_metadata_text,
    extract_doi,
    extract_embedded_metadata,
    extract_metadata,
    extract_text_from_first_pages,
    extract_year,
    first_nonempty_lines,
    guess_authors_from_first_page,
    guess_metadata_from_filename,
    guess_title_from_first_page,
    lookup_crossref_metadata,
    looks_like_bad_title,
    looks_like_header_noise,
    merge_metadata,
    metadata_from_crossref_work,
)
from kurrent.schema import ExtractedMetadata


def write_text_pdf(
    path: Path,
    pages: list[str],
    metadata: dict | None = None,
) -> Path:
    pdf = pymupdf.open()

    if metadata is not None:
        pdf.set_metadata(metadata)

    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)

    pdf.save(path)
    pdf.close()

    return path


def test_clean_metadata_text_normalizes_whitespace():
    """Verify that metadata strings are stripped and whitespace-normalized."""

    assert clean_metadata_text("  Still\nBuilding   the Memex  ") == (
        "Still Building the Memex"
    )


def test_clean_metadata_text_returns_none_for_empty_values():
    """Verify that missing or blank metadata fields become None."""

    assert clean_metadata_text(None) is None
    assert clean_metadata_text("") is None
    assert clean_metadata_text("   \n\t  ") is None




def test_clean_author_metadata_text_name_cases_all_caps_authors():
    """Verify that obvious ALL-CAPS author metadata is made readable."""

    assert clean_author_metadata_text(
        "WILLIAM W. COHEN and YORAM SINGER"
    ) == "William W. Cohen and Yoram Singer"


def test_clean_author_metadata_text_leaves_mixed_case_authors_alone():
    """Verify that mixed-case author metadata is not re-cased."""

    assert clean_author_metadata_text(
        "William W. Cohen and Yoram Singer"
    ) == "William W. Cohen and Yoram Singer"

def test_clean_title_metadata_text_title_cases_all_caps_titles():
    """Verify that obvious ALL-CAPS titles are made readable."""

    assert clean_title_metadata_text(
        "THE EVOLUTION OF COOPERATION IN COMPLEX NETWORKS"
    ) == "The Evolution of Cooperation in Complex Networks"


def test_clean_title_metadata_text_leaves_mixed_case_titles_alone():
    """Verify that mixed-case titles are not re-cased."""

    assert clean_title_metadata_text(
        "Still Building the Memex"
    ) == "Still Building the Memex"


def test_metadata_from_crossref_work_cleans_all_caps_metadata():
    """Verify that Crossref ALL-CAPS titles and authors are normalized."""

    metadata = metadata_from_crossref_work({
        "title": ["THE EVOLUTION OF COOPERATION"],
        "author": [
            {"given": "MARTIN", "family": "NOWAK"},
            {"given": "ROBERT", "family": "MAY"},
        ],
        "issued": {"date-parts": [[2006]]},
        "DOI": "10.123/example",
    })

    assert metadata.title == "The Evolution of Cooperation"
    assert metadata.authors == "Martin Nowak, Robert May"


def test_looks_like_bad_title_rejects_common_pdf_metadata_junk():
    """Verify that common non-title PDF metadata values are rejected."""

    assert looks_like_bad_title(None)
    assert looks_like_bad_title("Microsoft Word - manuscript_final.docx")
    assert looks_like_bad_title("untitled")
    assert looks_like_bad_title("paper.pdf")


def test_looks_like_bad_title_accepts_plausible_title():
    """Verify that plausible embedded titles are accepted."""

    assert not looks_like_bad_title("Still Building the Memex")


def test_extract_doi_finds_first_doi_like_string():
    """Verify that DOI extraction finds DOI-like strings in text."""

    text = "review articles doi:10.1145/1897816.1897840 by Stephen Davies"

    assert extract_doi(text) == "10.1145/1897816.1897840"


def test_extract_doi_strips_trailing_punctuation():
    """Verify that DOI extraction trims common sentence punctuation."""

    text = "The DOI is 10.1145/1897816.1897840."

    assert extract_doi(text) == "10.1145/1897816.1897840"


def test_extract_doi_returns_none_when_absent():
    """Verify that DOI extraction returns None when no DOI is present."""

    assert extract_doi("No DOI appears in this text.") is None


def test_extract_year_prefers_publication_month_year_pattern():
    """Verify that publication-looking dates beat older cited years.

    The text mentions 1945, but the publication header says February 2011.
    """
    text = (
        "communications of the acm | february 2011 | vol. 54 | no. 2\n"
        "The information overload problem was already formidable in 1945."
    )

    assert extract_year(text) == 2011


def test_extract_year_falls_back_to_later_plausible_year():
    """Verify that generic year extraction avoids earliest-year bias."""

    text = "Bush wrote in 1945. This article was published in 2011."

    assert extract_year(text) == 2011


def test_extract_year_returns_none_when_absent():
    """Verify that year extraction returns None when no year is present."""

    assert extract_year("No publication date appears here.") is None


def test_first_nonempty_lines_returns_clean_early_lines():
    """Verify that blank lines are skipped and remaining lines are cleaned."""

    text = "\n\n  First line  \n\nSecond   line\nThird line"

    assert first_nonempty_lines(text, limit=2) == [
        "First line",
        "Second line",
    ]


def test_looks_like_header_noise_rejects_common_non_title_lines():
    """Verify that obvious first-page noise is not treated as a title."""

    assert looks_like_header_noise("doi:10.1145/1897816.1897840")
    assert looks_like_header_noise("https://example.com/paper")
    assert looks_like_header_noise("Abstract")
    assert looks_like_header_noise("© 2011 ACM")


def test_guess_title_from_first_page_skips_noise():
    """Verify that title guessing skips DOI/header noise."""

    text = """
    doi:10.1145/1897816.1897840
    review articles
    Still Building the Memex
    by Stephen Davies
    """

    assert guess_title_from_first_page(text) == "review articles"


def test_guess_authors_from_first_page_uses_line_after_title():
    """Verify that author guessing uses the line after the guessed title."""

    text = """
    Still Building the Memex
    by Stephen Davies
    Abstract
    This article discusses personal knowledge bases.
    """

    assert guess_authors_from_first_page(
        text,
        "Still Building the Memex",
    ) == "by Stephen Davies"


def test_guess_authors_from_first_page_rejects_affiliation_line():
    """Verify that affiliation-like lines are not accepted as authors."""

    text = """
    Still Building the Memex
    University of Mary Washington
    Abstract
    """

    assert guess_authors_from_first_page(
        text,
        "Still Building the Memex",
    ) is None


def test_guess_metadata_from_filename_extracts_year_and_title():
    """Verify that filename fallback extracts a year and cleaned title."""

    metadata = guess_metadata_from_filename(
        Path("/tmp/Epstein_2006_Generative_Social_Science.pdf")
    )

    assert metadata == ExtractedMetadata(
        title="Epstein Generative Social Science",
        year=2006,
    )


def test_merge_metadata_fills_only_missing_fields():
    """Verify that merge_metadata preserves primary values when present."""

    primary = ExtractedMetadata(
        title="Primary Title",
        authors=None,
        year=None,
        doi="10.1/primary",
    )
    fallback = ExtractedMetadata(
        title="Fallback Title",
        authors="Fallback Author",
        year=2011,
        doi="10.1/fallback",
    )

    assert merge_metadata(primary, fallback) == ExtractedMetadata(
        title="Primary Title",
        authors="Fallback Author",
        year=2011,
        doi="10.1/primary",
    )


def test_extract_embedded_metadata_reads_pdf_title_and_author(tmp_path):
    """Verify that embedded PDF title and author metadata can be extracted."""

    pdf_path = write_text_pdf(
        tmp_path / "paper.pdf",
        ["This is the first page."],
        metadata={
            "title": "Still Building the Memex",
            "author": "Stephen Davies",
        },
    )

    assert extract_embedded_metadata(pdf_path) == ExtractedMetadata(
        title="Still Building the Memex",
        authors="Stephen Davies",
    )


def test_extract_embedded_metadata_rejects_bad_title(tmp_path):
    """Verify that bad embedded titles are discarded but authors remain."""

    pdf_path = write_text_pdf(
        tmp_path / "paper.pdf",
        ["This is the first page."],
        metadata={
            "title": "Microsoft Word - manuscript_final.docx",
            "author": "Stephen Davies",
        },
    )

    assert extract_embedded_metadata(pdf_path) == ExtractedMetadata(
        title=None,
        authors="Stephen Davies",
    )


def test_extract_text_from_first_pages_respects_page_limit(tmp_path):
    """Verify that only the requested number of early pages is extracted."""

    pdf_path = write_text_pdf(
        tmp_path / "paper.pdf",
        [
            "This is page one.",
            "This is page two.",
            "This is page three.",
        ],
    )

    text = extract_text_from_first_pages(pdf_path, max_pages=2)

    assert "This is page one." in text
    assert "This is page two." in text
    assert "This is page three." not in text


def test_extract_metadata_combines_embedded_text_and_filename_sources(tmp_path):
    """Verify that extract_metadata combines several local metadata signals."""

    pdf_path = write_text_pdf(
        tmp_path / "Davies_2011_Still_Building_the_Memex.pdf",
        [
            "communications of the acm | february 2011 | vol. 54 | no. 2\n"
            "review articles\n"
            "doi:10.1145/1897816.1897840\n"
            "Still Building the Memex\n"
            "by Stephen Davies\n"
            "As World War II mercifully drew to a close in 1945..."
        ],
        metadata={
            "title": "Microsoft Word - manuscript_final.docx",
            "author": "",
        },
    )

    metadata = extract_metadata(pdf_path)

    assert metadata.year == 2011
    assert metadata.doi == "10.1145/1897816.1897840"
    assert metadata.title is not None


def test_metadata_from_crossref_work_normalizes_fields():
    """Verify that Crossref work JSON becomes ExtractedMetadata."""

    work = {
        "title": ["Still Building the Memex"],
        "author": [
            {"given": "Stephen", "family": "Davies"},
            {"family": "Solo"},
        ],
        "published-print": {"date-parts": [[2011, 2]]},
        "DOI": "10.1145/1897816.1897840",
    }

    assert metadata_from_crossref_work(work) == ExtractedMetadata(
        title="Still Building the Memex",
        authors="Stephen Davies, Solo",
        year=2011,
        doi="10.1145/1897816.1897840",
    )


def test_lookup_crossref_metadata_returns_empty_metadata_on_error(monkeypatch):
    """Verify that Crossref lookup failures degrade gracefully."""

    def fake_urlopen(request, timeout=10.0):
        raise TimeoutError("too slow")

    monkeypatch.setattr(
        "kurrent.metadata_extractor.urlopen",
        fake_urlopen,
    )

    assert lookup_crossref_metadata("10.1234/example") == ExtractedMetadata()


def test_extract_metadata_uses_crossref_when_doi_lookup_enabled(
    tmp_path,
    monkeypatch,
):
    """Verify that DOI lookup metadata wins over local guesses."""

    pdf_path = write_text_pdf(
        tmp_path / "local_guess_1945.pdf",
        [
            "doi:10.1145/1897816.1897840\n"
            "Local Guess Title\n"
            "by Local Author\n"
            "Bush wrote about the memex in 1945."
        ],
    )

    def fake_lookup(doi, crossref_mailto=None):
        return ExtractedMetadata(
            title="Still Building the Memex",
            authors="Stephen Davies",
            year=2011,
            doi=doi,
        )

    monkeypatch.setattr(
        "kurrent.metadata_extractor.lookup_crossref_metadata",
        fake_lookup,
    )

    metadata = extract_metadata(
        pdf_path,
        doi_lookup=True,
        crossref_mailto="stephen@example.edu",
    )

    assert metadata == ExtractedMetadata(
        title="Still Building the Memex",
        authors="Stephen Davies",
        year=2011,
        doi="10.1145/1897816.1897840",
    )


def test_extract_metadata_skips_crossref_when_doi_lookup_disabled(
    tmp_path,
    monkeypatch,
):
    """Verify that DOI lookup is opt-in."""

    pdf_path = write_text_pdf(
        tmp_path / "local_guess.pdf",
        [
            "doi:10.1145/1897816.1897840\n"
            "Local Guess Title\n"
            "by Local Author"
        ],
    )

    def fake_lookup(doi, crossref_mailto=None):
        raise AssertionError("Crossref lookup should not be called")

    monkeypatch.setattr(
        "kurrent.metadata_extractor.lookup_crossref_metadata",
        fake_lookup,
    )

    metadata = extract_metadata(pdf_path, doi_lookup=False)

    assert metadata.doi == "10.1145/1897816.1897840"
