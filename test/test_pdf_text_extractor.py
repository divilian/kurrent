from kurrent.pdf_text_extractor import (
    PageExtraction,
    WordBox,
    extract_page_from_words,
    find_likely_column_gutter,
    group_words_into_visual_rows,
)


def word(
    text: str,
    x0: float,
    y0: float,
    x1: float | None = None,
    y1: float | None = None,
) -> WordBox:
    """Create a fake word box for layout extraction tests."""

    if x1 is None:
        x1 = x0 + max(10, len(text) * 5)

    if y1 is None:
        y1 = y0 + 10

    return WordBox(
        page=1,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
    )


def two_column_words() -> list[WordBox]:
    """Return enough fake words for the gutter detector to see two columns."""

    words: list[WordBox] = []

    for row in range(20):
        y = 100 + row * 14
        words.extend(
            [
                word(f"L{row}a", 60, y, 90, y + 10),
                word(f"L{row}b", 100, y, 130, y + 10),
                word(f"R{row}a", 330, y, 360, y + 10),
                word(f"R{row}b", 370, y, 400, y + 10),
            ]
        )

    return words


def test_group_words_into_visual_rows_uses_y_position_not_input_order():
    """Verify that words are reconstructed into visual rows by coordinates."""

    words = [
        word("second", 60, 120),
        word("row", 110, 120),
        word("first", 60, 100),
        word("row", 105, 100),
    ]

    rows = group_words_into_visual_rows(words)
    row_texts = [" ".join(item.text for item in row) for row in rows]

    assert row_texts == ["first row", "second row"]


def test_find_likely_column_gutter_detects_center_gap():
    """Verify that a persistent center gap is detected as a column gutter."""

    gutter_x = find_likely_column_gutter(
        two_column_words(),
        page_width=500,
        page_height=700,
    )

    assert gutter_x is not None
    assert 200 < gutter_x < 300


def test_two_column_lines_are_ordered_left_column_then_right_column():
    """Verify that simultaneous left/right visual rows are not interleaved."""

    page = extract_page_from_words(
        words=two_column_words(),
        page_number=1,
        page_width=500,
        page_height=700,
    )

    line_texts = [line.text for line in page.lines]

    assert page.layout == "two-column"
    assert line_texts[:3] == [
        "L0a L0b",
        "L1a L1b",
        "L2a L2b",
    ]
    assert line_texts[20:23] == [
        "R0a R0b",
        "R1a R1b",
        "R2a R2b",
    ]


def test_full_width_heading_breaks_column_bands():
    """Verify full-width lines are kept between two-column reading bands."""

    words = []

    for row in range(20):
        y = 100 + row * 14
        words.extend(
            [
                word(f"A{row}L", 60, y, 95, y + 10),
                word(f"A{row}R", 340, y, 375, y + 10),
            ]
        )

    words.extend(
        [
            word("Full", 180, 400, 210, 410),
            word("Heading", 216, 400, 330, 410),
        ]
    )

    for row in range(20):
        y = 450 + row * 14
        words.extend(
            [
                word(f"B{row}L", 60, y, 95, y + 10),
                word(f"B{row}R", 340, y, 375, y + 10),
            ]
        )

    page = extract_page_from_words(
        words=words,
        page_number=1,
        page_width=500,
        page_height=800,
    )
    line_texts = [line.text for line in page.lines]

    assert "Full Heading" in line_texts
    heading_index = line_texts.index("Full Heading")
    assert line_texts.index("A0L") < heading_index
    assert line_texts.index("A0R") < heading_index
    assert heading_index < line_texts.index("B0L")
    assert heading_index < line_texts.index("B0R")

from kurrent.pdf_text_extractor import (
    TextLine,
    filter_boilerplate_lines,
    filter_margin_artifact_words,
)


def test_gutter_detection_handles_two_column_body_after_full_width_front_matter():
    """Verify two-column body can be detected below full-width front matter."""

    words: list[WordBox] = []

    for row in range(18):
        y = 70 + row * 14
        words.extend(
            [
                word(f"Abstract{row}a", 100, y, 170, y + 10),
                word(f"Abstract{row}b", 180, y, 250, y + 10),
                word(f"Abstract{row}c", 260, y, 330, y + 10),
                word(f"Abstract{row}d", 340, y, 410, y + 10),
            ]
        )

    for row in range(8):
        y = 390 + row * 14
        words.extend(
            [
                word(f"L{row}a", 60, y, 90, y + 10),
                word(f"L{row}b", 100, y, 130, y + 10),
                word(f"R{row}a", 330, y, 360, y + 10),
                word(f"R{row}b", 370, y, 400, y + 10),
            ]
        )

    page = extract_page_from_words(
        words=words,
        page_number=1,
        page_width=500,
        page_height=700,
    )
    line_texts = [line.text for line in page.lines]

    assert page.layout == "two-column"
    assert line_texts.index("L0a L0b") < line_texts.index("R0a R0b")
    assert line_texts.index("L7a L7b") < line_texts.index("R0a R0b")


def test_margin_artifact_filter_removes_tall_edge_words():
    """Verify rotated margin boilerplate words are removed before line grouping."""

    words = [
        word("NIH-PA", 20, 100, 33, 145),
        word("Author", 20, 150, 33, 190),
        word("Real", 100, 100, 130, 110),
        word("content", 136, 100, 190, 110),
    ]

    filtered = filter_margin_artifact_words(
        words,
        page_width=600,
        page_height=800,
    )

    assert [item.text for item in filtered] == ["Real", "content"]


def test_boilerplate_filter_removes_copyright_notice():
    """Verify publisher copyright lines are filtered out of body text."""

    lines = [
        TextLine(
            page=1,
            text="Copyright © 2025, Association for the Advancement of Artificial",
            x0=50,
            y0=100,
            x1=400,
            y1=110,
            column="left",
        ),
        TextLine(
            page=1,
            text="Intelligence (www.aaai.org). All rights reserved.",
            x0=50,
            y0=112,
            x1=400,
            y1=122,
            column="left",
        ),
        TextLine(
            page=1,
            text="the behavioral dynamics that may arise from the interaction",
            x0=50,
            y0=124,
            x1=400,
            y1=134,
            column="left",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == [
        "the behavioral dynamics that may arise from the interaction"
    ]

from kurrent.pdf_text_extractor import filter_repeated_margin_lines


def test_boilerplate_filter_removes_pacs_number_line():
    """Verify PACS metadata is filtered out of body text."""

    lines = [
        TextLine(
            page=1,
            text="PACS number s : 89.75.Hc, 87.23.Kg, 02.50.Le",
            x0=50,
            y0=100,
            x1=400,
            y1=110,
            column="right",
        ),
        TextLine(
            page=1,
            text="Furthermore, it has been found that cooperation increases",
            x0=50,
            y0=112,
            x1=400,
            y1=122,
            column="right",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == [
        "Furthermore, it has been found that cooperation increases"
    ]


def test_repeated_margin_filter_removes_running_header_lines():
    """Verify repeated top-margin headers are removed across pages."""

    pages = []

    for page_number in range(1, 4):
        pages.append(
            PageExtraction(
                page=page_number,
                width=600,
                height=800,
                layout="single-column",
                gutter_x=None,
                lines=[
                    TextLine(
                        page=page_number,
                        text="Fu et al.",
                        x0=80,
                        y0=20,
                        x1=130,
                        y1=30,
                        column="single",
                    ),
                    TextLine(
                        page=page_number,
                        text=f"Page {page_number}",
                        x0=500,
                        y0=20,
                        x1=550,
                        y1=30,
                        column="single",
                    ),
                    TextLine(
                        page=page_number,
                        text=f"real body line {page_number}",
                        x0=80,
                        y0=120,
                        x1=300,
                        y1=130,
                        column="single",
                    ),
                ],
            )
        )

    filtered_pages = filter_repeated_margin_lines(pages)

    assert [[line.text for line in page.lines] for page in filtered_pages] == [
        ["real body line 1"],
        ["real body line 2"],
        ["real body line 3"],
    ]
    assert [line.text for line in filtered_pages[1].filtered_lines] == [
        "Fu et al.",
        "Page 2",
    ]


def test_repeated_margin_filter_keeps_repeated_body_text():
    """Verify repeated text is not removed when it appears in the body zone."""

    pages = []

    for page_number in range(1, 4):
        pages.append(
            PageExtraction(
                page=page_number,
                width=600,
                height=800,
                layout="single-column",
                gutter_x=None,
                lines=[
                    TextLine(
                        page=page_number,
                        text="Methods",
                        x0=80,
                        y0=350,
                        x1=140,
                        y1=360,
                        column="single",
                    ),
                ],
            )
        )

    filtered_pages = filter_repeated_margin_lines(pages)

    assert [[line.text for line in page.lines] for page in filtered_pages] == [
        ["Methods"],
        ["Methods"],
        ["Methods"],
    ]

from kurrent.pdf_text_extractor import normalize_extracted_text


def test_normalize_extracted_text_repairs_control_glyph_punctuation():
    """Verify custom-encoded PDF punctuation is repaired before cleanup."""

    text = "needed \x01see Ref. \x032\x04 for a recent review\x02."

    assert normalize_extracted_text(text) == (
        "needed (see Ref. [2] for a recent review)."
    )


def test_boilerplate_filter_removes_bare_copyright_symbol_line():
    """Verify copyright-symbol publisher notices are filtered."""

    lines = [
        TextLine(
            page=1,
            text="©2009 The American Physical Society",
            x0=400,
            y0=760,
            x1=560,
            y1=770,
            column="right",
        ),
        TextLine(
            page=1,
            text="However, the situation",
            x0=310,
            y0=740,
            x1=450,
            y1=750,
            column="right",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == ["However, the situation"]


def test_repeated_margin_filter_keeps_unique_short_bottom_body_line():
    """Verify short bottom body lines are not dropped merely for position."""

    pages = [
        PageExtraction(
            page=1,
            width=600,
            height=800,
            layout="single-column",
            gutter_x=None,
            lines=[
                TextLine(
                    page=1,
                    text="system of three coupled ordinary differential equations as",
                    x0=310,
                    y0=725,
                    x1=560,
                    y1=736,
                    column="right",
                ),
                TextLine(
                    page=1,
                    text="follows:",
                    x0=310,
                    y0=737,
                    x1=345,
                    y1=748,
                    column="right",
                ),
            ],
        )
    ]

    filtered_pages = filter_repeated_margin_lines(pages)

    assert [line.text for line in filtered_pages[0].lines] == [
        "system of three coupled ordinary differential equations as",
        "follows:",
    ]


def test_repeated_margin_filter_keeps_unique_top_section_heading():
    """Verify non-repeated top-zone headings are not treated as headers."""

    pages = [
        PageExtraction(
            page=1,
            width=600,
            height=800,
            layout="single-column",
            gutter_x=None,
            lines=[
                TextLine(
                    page=1,
                    text="1 Introduction",
                    x0=80,
                    y0=45,
                    x1=190,
                    y1=56,
                    column="single",
                ),
                TextLine(
                    page=1,
                    text="Large language models can operate as social agents.",
                    x0=80,
                    y0=85,
                    x1=420,
                    y1=96,
                    column="single",
                ),
            ],
        )
    ]

    filtered_pages = filter_repeated_margin_lines(pages)

    assert [line.text for line in filtered_pages[0].lines] == [
        "1 Introduction",
        "Large language models can operate as social agents.",
    ]
    assert filtered_pages[0].filtered_lines == []


def test_repeated_margin_filter_keeps_unique_bottom_body_sentence():
    """Verify non-repeated bottom-zone prose is not dropped as a footer."""

    pages = [
        PageExtraction(
            page=1,
            width=600,
            height=800,
            layout="single-column",
            gutter_x=None,
            lines=[
                TextLine(
                    page=1,
                    text="This result completes the proof of the theorem.",
                    x0=80,
                    y0=725,
                    x1=390,
                    y1=736,
                    column="single",
                ),
            ],
        )
    ]

    filtered_pages = filter_repeated_margin_lines(pages)

    assert [line.text for line in filtered_pages[0].lines] == [
        "This result completes the proof of the theorem."
    ]
    assert filtered_pages[0].filtered_lines == []


def test_boilerplate_filter_keeps_body_sentence_containing_copyright():
    """Verify ordinary prose mentioning copyright is not filtered."""

    lines = [
        TextLine(
            page=1,
            text="The copyright status of these documents affects reuse.",
            x0=80,
            y0=250,
            x1=420,
            y1=261,
            column="single",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == [
        "The copyright status of these documents affects reuse."
    ]


def test_boilerplate_filter_keeps_body_sentence_containing_doi():
    """Verify DOI is filtered only as metadata, not as ordinary prose."""

    lines = [
        TextLine(
            page=1,
            text="We use DOI metadata only to improve document lookup.",
            x0=80,
            y0=250,
            x1=420,
            y1=261,
            column="single",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == [
        "We use DOI metadata only to improve document lookup."
    ]


def test_compact_fu2009_page_filters_pacs_and_keeps_column_order():
    """Regression-style check for fu2009-like two-column cleanup."""

    words: list[WordBox] = []

    for row in range(12):
        y = 100 + row * 14
        words.extend(
            [
                word(f"left{row}a", 60, y, 100, y + 10),
                word(f"left{row}b", 108, y, 148, y + 10),
                word(f"right{row}a", 330, y, 378, y + 10),
                word(f"right{row}b", 386, y, 434, y + 10),
            ]
        )

    words.extend(
        [
            word("PACS", 330, 282, 360, 292),
            word("number", 366, 282, 420, 292),
            word("s", 424, 282, 430, 292),
            word(":", 434, 282, 438, 292),
            word("89.75.Hc", 444, 282, 500, 292),
        ]
    )

    page = extract_page_from_words(
        words=words,
        page_number=1,
        page_width=560,
        page_height=760,
    )
    line_texts = [line.text for line in page.lines]
    filtered_texts = [line.text for line in page.filtered_lines]

    assert page.layout == "two-column"
    assert line_texts.index("left0a left0b") < line_texts.index("right0a right0b")
    assert "PACS number s: 89.75.Hc" not in line_texts
    assert "PACS number s: 89.75.Hc" in filtered_texts


def test_compact_fu2008_page_filters_running_head_but_keeps_body():
    """Regression-style check for fu2008-like running headers."""

    pages = []

    for page_number in range(1, 4):
        pages.append(
            PageExtraction(
                page=page_number,
                width=600,
                height=800,
                layout="single-column",
                gutter_x=None,
                lines=[
                    TextLine(
                        page=page_number,
                        text="Fu et al.",
                        x0=80,
                        y0=20,
                        x1=125,
                        y1=31,
                        column="single",
                    ),
                    TextLine(
                        page=page_number,
                        text=f"Page {page_number}",
                        x0=500,
                        y0=20,
                        x1=545,
                        y1=31,
                        column="single",
                    ),
                    TextLine(
                        page=page_number,
                        text="employed for investigating the origin of cooperation [4].",
                        x0=80,
                        y0=95,
                        x1=450,
                        y1=106,
                        column="single",
                    ),
                ],
            )
        )

    filtered_pages = filter_repeated_margin_lines(pages)

    assert [line.text for line in filtered_pages[1].lines] == [
        "employed for investigating the origin of cooperation [4]."
    ]
    assert [line.text for line in filtered_pages[1].filtered_lines] == [
        "Fu et al.",
        "Page 2",
    ]


def test_compact_fontana_page_filters_copyright_but_keeps_following_body():
    """Regression-style check for fontana2025-like copyright cleanup."""

    lines = [
        TextLine(
            page=1,
            text="tools (Fontana et al. 2025). To understand and anticipate",
            x0=60,
            y0=480,
            x1=285,
            y1=491,
            column="left",
        ),
        TextLine(
            page=1,
            text="Copyright © 2025, Association for the Advancement of Artificial",
            x0=60,
            y0=500,
            x1=285,
            y1=511,
            column="left",
        ),
        TextLine(
            page=1,
            text="Intelligence (www.aaai.org). All rights reserved.",
            x0=60,
            y0=514,
            x1=285,
            y1=525,
            column="left",
        ),
        TextLine(
            page=1,
            text="the behavioral dynamics that may arise from the interac-",
            x0=60,
            y0=540,
            x1=285,
            y1=551,
            column="left",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == [
        "tools (Fontana et al. 2025). To understand and anticipate",
        "the behavioral dynamics that may arise from the interac-",
    ]
