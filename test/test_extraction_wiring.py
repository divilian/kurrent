from types import SimpleNamespace

from kurrent import chunker, sectioner
from kurrent.schema import SectionLine


def test_sectioner_extracts_lines_through_layout_aware_extractor(monkeypatch):
    """Verify sectioner uses pdf_text_extractor instead of raw PyMuPDF text."""

    def fake_extract_layout_pdf_lines(_pdf_path):
        return [
            SectionLine(page=2, text="  First layout-aware line  "),
            SectionLine(page=3, text="Second layout-aware line"),
        ]

    monkeypatch.setattr(
        sectioner,
        "extract_layout_pdf_lines",
        fake_extract_layout_pdf_lines,
    )

    lines = sectioner._extract_pdf_lines_with_pages("fake.pdf")

    assert lines == [
        (2, "First layout-aware line"),
        (3, "Second layout-aware line"),
    ]


def test_sectioner_heading_detection_uses_layout_aware_extractor(monkeypatch):
    """Verify rules-based candidate detection uses layout-aware lines."""

    def fake_extract_layout_pdf_lines(_pdf_path):
        return [
            SectionLine(page=1, text="1 Introduction"),
            SectionLine(page=2, text="2 Methods"),
            SectionLine(page=9, text="3 Results"),
        ]

    monkeypatch.setattr(
        sectioner,
        "extract_layout_pdf_lines",
        fake_extract_layout_pdf_lines,
    )

    candidates = sectioner.detect_heading_candidates("fake.pdf", max_pages=8)

    assert candidates == ["1 Introduction", "2 Methods"]


def test_chunker_legacy_page_extraction_uses_layout_aware_extractor(monkeypatch):
    """Verify chunker.extract_pdf_pages no longer bypasses the extractor."""

    def fake_extract_layout_pdf_pages(_pdf_path):
        return [
            SimpleNamespace(
                page=1,
                lines=[
                    SimpleNamespace(text="first layout line"),
                    SimpleNamespace(text="second layout line"),
                ],
            ),
            SimpleNamespace(
                page=2,
                lines=[SimpleNamespace(text="third layout line")],
            ),
        ]

    monkeypatch.setattr(
        chunker,
        "extract_layout_pdf_pages",
        fake_extract_layout_pdf_pages,
    )

    pages = chunker.extract_pdf_pages("fake.pdf")

    assert pages == {
        1: "first layout line second layout line",
        2: "third layout line",
    }
