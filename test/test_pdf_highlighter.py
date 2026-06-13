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


def test_long_fallback_excerpt_is_not_highlighted_directly_when_selector_fails(tmp_path):
    """Whole-chunk fallback text should not highlight page headers/title wholesale."""

    pdf_path = tmp_path / "paper.pdf"
    make_pdf_with_text(
        pdf_path,
        "review articles journal header Article Title By Author. "
        "This relevant paragraph is the passage that should be highlighted.",
    )

    long_fallback = " ".join(["review articles journal header Article Title By Author"] * 30)

    result = pdf_highlighter.create_highlighted_pdf_for_research_interest(
        pdf_path=pdf_path,
        page_start=1,
        research_interest="knowledge bases",
        fallback_excerpt=long_fallback,
        excerpt_selector=lambda page_text, query: None,
        output_dir=tmp_path / "highlights",
    )

    assert result.success is False
    assert result.highlighted_pdf_path is None
    assert "No relevant excerpt" in result.message


def test_ollama_prompt_anchors_highlight_to_specific_evidence_excerpt():
    """The default LLM selector should receive the specific source evidence."""

    messages = pdf_highlighter._ollama_excerpt_messages(
        page_text="Page paragraph one. Page paragraph two.",
        research_interest="personal knowledge bases",
        evidence_excerpt="Specific evidence chunk for passage 1b.",
    )

    prompt = messages[1]["content"]
    assert "Retrieved evidence excerpt for this specific source item" in prompt
    assert "Specific evidence chunk for passage 1b" in prompt
    assert "best corresponds to the retrieved evidence excerpt" in prompt


def test_create_metadata_highlighted_pdf_colors_proposed_fields(tmp_path):
    """Metadata review highlighting should annotate a temp copy, not the source."""

    from kurrent.schema import ExtractedMetadata

    pdf_path = tmp_path / "metadata.pdf"
    make_pdf_with_text(
        pdf_path,
        "Natural-Language Multi-Agent Simulations of\n"
        "Argumentative Opinion Dynamics\n"
        "Gregor Betz\n"
        "Preprint 2021 doi 10.1234/example",
    )

    metadata = ExtractedMetadata(
        title="Natural-Language Multi-Agent Simulations of Argumentative Opinion Dynamics",
        authors="Gregor Betz",
        year=2021,
        doi="10.1234/example",
    )

    result = pdf_highlighter.create_metadata_highlighted_pdf(
        pdf_path,
        metadata,
        output_dir=tmp_path / "metadata-highlights",
    )

    assert result.success is True
    assert result.highlighted_pdf_path is not None
    assert result.highlighted_pdf_path.exists()
    assert result.highlighted_pdf_path != pdf_path

    source_doc = fitz.open(pdf_path)
    highlighted_doc = fitz.open(result.highlighted_pdf_path)
    try:
        assert list(source_doc[0].annots() or []) == []
        annotations = list(highlighted_doc[0].annots() or [])
        colors = [annot.colors.get("stroke") for annot in annotations]
    finally:
        source_doc.close()
        highlighted_doc.close()

    rounded_colors = {tuple(round(channel, 2) for channel in color) for color in colors}

    assert len(annotations) >= 4
    assert (1.0, 0.0, 0.0) in rounded_colors
    assert (1.0, 0.55, 0.0) in rounded_colors
    assert (0.0, 0.25, 1.0) in rounded_colors
    assert (1.0, 0.35, 0.8) in rounded_colors
