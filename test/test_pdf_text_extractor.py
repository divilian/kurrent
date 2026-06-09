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


def test_two_column_ordering_is_column_first_not_band_first():
    """Verify right-column text does not interrupt a left-column section."""

    words = []

    for row, text in enumerate([
        "1 INTRODUCTION",
        "The Prisoners Dilemma game",
        "studies cooperation",
    ]):
        y = 100 + row * 14
        x = 60
        for word_text in text.split():
            words.append(word(word_text, x, y, x + len(word_text) * 6, y + 10))
            x += len(word_text) * 6 + 5

    for row, text in enumerate([
        "neighborhood is NS",
        "game playing neighborhood",
        "strategy update neighborhood",
    ]):
        y = 100 + row * 14
        x = 330
        for word_text in text.split():
            words.append(word(word_text, x, y, x + len(word_text) * 6, y + 10))
            x += len(word_text) * 6 + 5

    # Add enough lower body rows for robust two-column detection.
    for row in range(12):
        y = 170 + row * 14
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
    assert line_texts.index("1 INTRODUCTION") < line_texts.index("The Prisoners Dilemma game")
    assert line_texts.index("studies cooperation") < line_texts.index("neighborhood is NS")

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

from kurrent.pdf_text_extractor import dehyphenate_line_breaks


def test_dehyphenate_line_breaks_joins_split_words():
    """Verify line-final hyphenation is repaired for ordinary split words."""

    lines = [
        TextLine(
            page=1,
            text="LLMs can operate as artifi-",
            x0=60,
            y0=100,
            x1=250,
            y1=111,
            column="left",
        ),
        TextLine(
            page=1,
            text="cial social agents.",
            x0=60,
            y0=114,
            x1=250,
            y1=125,
            column="left",
        ),
    ]

    repaired = dehyphenate_line_breaks(lines)

    assert [line.text for line in repaired] == [
        "LLMs can operate as artificial",
        "social agents.",
    ]


def test_dehyphenate_line_breaks_handles_short_prefix_fragments():
    """Verify short real word fragments such as co- / operation are repaired."""

    lines = [
        TextLine(
            page=1,
            text="the evolution of co-",
            x0=60,
            y0=100,
            x1=250,
            y1=111,
            column="left",
        ),
        TextLine(
            page=1,
            text="operation in networks",
            x0=60,
            y0=114,
            x1=250,
            y1=125,
            column="left",
        ),
    ]

    repaired = dehyphenate_line_breaks(lines)

    assert [line.text for line in repaired] == [
        "the evolution of cooperation",
        "in networks",
    ]


def test_dehyphenate_line_breaks_preserves_common_hyphenated_prefixes():
    """Verify likely real hyphenated compounds keep their hyphen."""

    lines = [
        TextLine(
            page=1,
            text="This produces a well-",
            x0=60,
            y0=100,
            x1=250,
            y1=111,
            column="left",
        ),
        TextLine(
            page=1,
            text="being effect.",
            x0=60,
            y0=114,
            x1=250,
            y1=125,
            column="left",
        ),
    ]

    repaired = dehyphenate_line_breaks(lines)

    assert [line.text for line in repaired] == [
        "This produces a well-being",
        "effect.",
    ]


def test_dehyphenate_line_breaks_does_not_join_before_uppercase_word():
    """Verify a hyphen before an uppercase starter is not treated as a split word."""

    lines = [
        TextLine(
            page=1,
            text="The options are A-",
            x0=60,
            y0=100,
            x1=250,
            y1=111,
            column="left",
        ),
        TextLine(
            page=1,
            text="Level and B-Level treatments.",
            x0=60,
            y0=114,
            x1=250,
            y1=125,
            column="left",
        ),
    ]

    repaired = dehyphenate_line_breaks(lines)

    assert [line.text for line in repaired] == [
        "The options are A-",
        "Level and B-Level treatments.",
    ]


def test_extract_page_from_words_dehyphenates_in_reading_stream():
    """Verify page extraction repairs split words after line ordering."""

    words = [
        word("artifi-", 60, 100, 105, 110),
        word("cial", 60, 114, 90, 124),
        word("agents", 96, 114, 140, 124),
    ]

    page = extract_page_from_words(
        words=words,
        page_number=1,
        page_width=500,
        page_height=700,
    )

    assert [line.text for line in page.lines] == [
        "artificial",
        "agents",
    ]


def test_normalize_extracted_text_replaces_common_ligatures():
    """Verify typographic ligatures are normalized for search and embedding."""

    assert normalize_extracted_text("ﬁxed ﬂow oﬀice aﬃnity") == (
        "fixed flow office affinity"
    )


def test_dehyphenate_line_breaks_does_not_join_across_columns():
    """Verify dehyphenation is restricted to compatible column geometry."""

    lines = [
        TextLine(
            page=1,
            text="network hetero-",
            x0=60,
            y0=100,
            x1=250,
            y1=111,
            column="left",
        ),
        TextLine(
            page=1,
            text="number of partnerships",
            x0=330,
            y0=100,
            x1=500,
            y1=111,
            column="right",
        ),
    ]

    repaired = dehyphenate_line_breaks(lines)

    assert [line.text for line in repaired] == [
        "network hetero-",
        "number of partnerships",
    ]


def test_figure_heavy_page_detects_columns_from_line_starts():
    """Verify figure-heavy pages still use left-column then right-column order."""

    words: list[WordBox] = []

    # Left column, including a split word that should be repaired before the
    # right column is emitted.
    for row, text in enumerate([
        "payoffs arising from network hetero-",
        "geneity would help the evolution of cooperation",
        "Moreover we confirm stabilization",
    ]):
        y = 300 + row * 14
        x = 60
        for word_text in text.split():
            words.append(word(word_text, x, y, x + len(word_text) * 4, y + 10))
            x += len(word_text) * 4 + 4

    # Right-column prose at similar y positions should not interrupt the left
    # column.
    for row, text in enumerate([
        "number of partnerships readily shift to cooperation during",
        "strategy dynamics avoiding further unfavorable outcomes",
    ]):
        y = 300 + row * 14
        x = 330
        for word_text in text.split():
            words.append(word(word_text, x, y, x + len(word_text) * 4, y + 10))
            x += len(word_text) * 4 + 4

    for row in range(12):
        y = 380 + row * 14
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
        page_width=560,
        page_height=760,
    )
    text = " ".join(line.text for line in page.lines)

    assert "network heterogeneity would help" in text
    assert "heteronumber" not in text
    assert text.index("network heterogeneity") < text.index("number of partnerships")


def test_first_page_author_blocks_stay_before_abstract_body():
    """Verify right-side author blocks do not interrupt left-column body text."""

    words: list[WordBox] = []

    # Full-width title.
    x = 60
    for word_text in "Prisoners Dilemma on Graphs".split():
        words.append(word(word_text, x, 70, x + len(word_text) * 6, 80))
        x += len(word_text) * 6 + 5

    # Two-column author/front-matter area above the abstract.
    for x0, text in [
        (90, "Left Author"),
        (330, "Right Author"),
    ]:
        x = x0
        for word_text in text.split():
            words.append(word(word_text, x, 115, x + len(word_text) * 6, 125))
            x += len(word_text) * 6 + 5

    # Left-column abstract and introduction body.
    for row, text in enumerate([
        "ABSTRACT",
        "abstract left body line",
        "1 INTRODUCTION",
        "Iterated prisoners dilemma and",
    ]):
        y = 200 + row * 14
        x = 60
        for word_text in text.split():
            words.append(word(word_text, x, y, x + len(word_text) * 6, y + 10))
            x += len(word_text) * 6 + 5

    # Right-column continuation of the introduction starts high on the page.
    for row, text in enumerate([
        "prisoners dilemma on graphs continues",
        "right column body line",
    ]):
        y = 210 + row * 14
        x = 330
        for word_text in text.split():
            words.append(word(word_text, x, y, x + len(word_text) * 6, y + 10))
            x += len(word_text) * 6 + 5

    # Add lower rows for robust two-column detection.
    for row in range(12):
        y = 330 + row * 14
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
        page_width=560,
        page_height=760,
    )
    line_texts = [line.text for line in page.lines]

    assert line_texts.index("Left Author") < line_texts.index("ABSTRACT")
    assert line_texts.index("Right Author") < line_texts.index("ABSTRACT")
    assert line_texts.index("Iterated prisoners dilemma and") < line_texts.index(
        "prisoners dilemma on graphs continues"
    )


def test_boilerplate_filter_removes_acm_first_page_notice():
    """Verify ACM conference copyright block lines are filtered."""

    lines = [
        TextLine(
            page=1,
            text="Copyright is held by the author/owner(s).",
            x0=60,
            y0=700,
            x1=260,
            y1=711,
            column="left",
        ),
        TextLine(
            page=1,
            text="GECCO’09, July 8–12, 2009, Montréal Québec, Canada.",
            x0=60,
            y0=712,
            x1=300,
            y1=723,
            column="left",
        ),
        TextLine(
            page=1,
            text="ACM 978-1-60558-505-5/09/07.",
            x0=60,
            y0=724,
            x1=230,
            y1=735,
            column="left",
        ),
        TextLine(
            page=1,
            text="prisoner’s dilemma on graphs continues",
            x0=330,
            y0=210,
            x1=500,
            y1=221,
            column="right",
        ),
    ]

    filtered = filter_boilerplate_lines(lines)

    assert [line.text for line in filtered] == [
        "prisoner’s dilemma on graphs continues"
    ]


def test_normalize_extracted_text_drops_lone_surrogate_codepoints():
    """Verify invalid Unicode surrogates cannot poison UTF-8 serialization."""

    from kurrent.pdf_text_extractor import normalize_extracted_text

    text = normalize_extracted_text("alpha \ud835 beta")

    assert text == "alpha beta"
    assert text.encode("utf-8") == b"alpha beta"
