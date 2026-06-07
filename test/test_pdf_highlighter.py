from pathlib import Path

import pytest

import kurrent.pdf_highlighter as pdf_highlighter

fitz = pytest.importorskip("fitz")


def make_pdf_with_text(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_fuzzy_match_excerpt_to_words_finds_exact_normalized_span(tmp_path):
    """Excerpt matching should map selected text back to PDF word coordinates."""

    pdf_path = tmp_path / "paper.pdf"
    make_pdf_with_text(
        pdf_path,
        "Alpha beta gamma delta. Homophily affects tie formation in networks.",
    )

    doc = fitz.open(pdf_path)
    try:
        words = pdf_highlighter._page_words(doc[0])
        match = pdf_highlighter.fuzzy_match_excerpt_to_words(
            words,
            "Homophily affects tie formation in networks",
        )
    finally:
        doc.close()

    assert match is not None
    assert match.method == "exact-token"
    assert match.score == 1.0
    matched_words = [word.normalized for word in words[match.start : match.end]]
    assert matched_words == [
        "homophily",
        "affects",
        "tie",
        "formation",
        "in",
        "networks",
    ]


def test_create_highlighted_pdf_for_research_interest_creates_temp_copy(tmp_path):
    """Highlighting should create a temporary annotated copy, not mutate the source."""

    pdf_path = tmp_path / "paper.pdf"
    out_dir = tmp_path / "highlights"
    make_pdf_with_text(
        pdf_path,
        "Alpha beta gamma delta. Homophily affects tie formation in networks.",
    )

    result = pdf_highlighter.create_highlighted_pdf_for_research_interest(
        pdf_path=pdf_path,
        page_start=1,
        research_interest="homophily and tie formation",
        excerpt_selector=lambda page_text, query: "Homophily affects tie formation in networks",
        output_dir=out_dir,
    )

    assert result.success is True
    assert result.page == 1
    assert result.highlighted_pdf_path is not None
    assert result.highlighted_pdf_path.exists()
    assert result.highlighted_pdf_path.parent == out_dir
    assert result.highlighted_pdf_path != pdf_path
    assert result.method == "exact-token"


def test_create_highlighted_pdf_fails_gracefully_when_excerpt_cannot_be_found(tmp_path):
    """Unlocatable selected text should return a failed result without raising."""

    pdf_path = tmp_path / "paper.pdf"
    make_pdf_with_text(pdf_path, "Alpha beta gamma delta.")

    result = pdf_highlighter.create_highlighted_pdf_for_research_interest(
        pdf_path=pdf_path,
        page_start=1,
        research_interest="homophily",
        excerpt_selector=lambda page_text, query: "Completely absent phrase",
        output_dir=tmp_path / "highlights",
    )

    assert result.success is False
    assert result.highlighted_pdf_path is None
    assert "could not be located" in result.message
