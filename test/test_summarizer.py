from pathlib import Path

import pytest

from kurrent import summarizer
from kurrent.schema import SectionSpan


def section(title, text, index=0):
    return SectionSpan(
        doc_id="doc",
        section_index=index,
        section_number=None,
        section_title=title,
        page_start=1,
        page_end=1,
        text=text,
        lines=None,
    )


def test_trim_section_for_summary_keeps_small_sections_whole():
    text = "This is a normal section."

    trimmed, was_truncated = summarizer.trim_section_for_summary(
        text,
        max_num_ctx=100,
    )

    assert trimmed == text
    assert was_truncated is False


def test_trim_section_for_summary_preserves_beginning_and_end_for_large_sections():
    text = "A" * 1000 + " middle " + "Z" * 1000

    trimmed, was_truncated = summarizer.trim_section_for_summary(
        text,
        max_num_ctx=100,
    )

    assert was_truncated is True
    assert trimmed.startswith("A" * 100)
    assert "Middle of this unusually long section omitted" in trimmed
    assert trimmed.endswith("Z" * 100)


def test_obvious_non_content_sections_are_skipped_but_related_work_is_content():
    assert summarizer.is_obvious_non_content_section(section("References", "x"))
    assert summarizer.is_obvious_non_content_section(section("Acknowledgments", "x"))
    assert not summarizer.is_obvious_non_content_section(section("Related Work", "x"))


def test_select_screening_sections_prefers_high_signal_sections_without_schema_assumption():
    sections = [
        section("front matter", "title block", index=-1),
        section("Abstract", "abstract text", index=0),
        section("Model", "model text", index=1),
        section("Results", "results text", index=2),
        section("Discussion and Future Work", "discussion text", index=3),
        section("References", "1. Someone. 2020.", index=4),
    ]

    selected = summarizer.select_screening_sections(sections, max_sections=3)

    assert [excerpt.section_title for excerpt in selected] == [
        "Abstract",
        "Model",
        "Discussion and Future Work",
    ]


def test_select_screening_sections_deduplicates_repeated_titles_and_text():
    sections = [
        section("Introduction", "intro text", index=0),
        section("Introduction", "second intro text", index=1),
        section("Model", "same text", index=2),
        section("Different Title", "same text", index=3),
        section("Conclusion", "conclusion text", index=4),
    ]

    selected = summarizer.select_screening_sections(sections, max_sections=10)

    assert [excerpt.section_title for excerpt in selected] == [
        "Introduction",
        "Model",
        "Conclusion",
    ]


def test_select_screening_sections_truncates_to_context_budget():
    sections = [section("Long", "A" * 2000 + "Z" * 2000, index=0)]

    selected = summarizer.select_screening_sections(
        sections,
        max_sections=1,
        max_num_ctx=200,
    )

    assert len(selected) == 1
    assert selected[0].was_truncated is True
    assert "Middle of this unusually long section omitted" in selected[0].text


def test_clean_screening_summary_text_does_not_force_extra_paragraphs():
    text = "Para one.\n\nPara two.\n\nPara three."

    normalized = summarizer.clean_screening_summary_text(text, 2)

    assert normalized == "Para one.\n\nPara two.\n\nPara three."




def test_clean_screening_summary_text_removes_obvious_markdown_symbols_but_keeps_lists():
    text = """### Major Point 1: Cooperation dynamics

1. **Model Description**: agents adapt their links.
2. **Steady States**: leaders sustain cooperation.

- Final note with `inline code`.
"""

    cleaned = summarizer.clean_screening_summary_text(text, 2)

    assert cleaned == (
        "Major Point 1: Cooperation dynamics\n\n"
        "1. Model Description: agents adapt their links.\n"
        "2. Steady States: leaders sustain cooperation.\n\n"
        "- Final note with inline code."
    )


def test_summarize_pdf_for_screening_uses_one_llm_call_over_selected_sections(monkeypatch):
    sections = [
        section("Abstract", "This paper studies cooperation.", index=0),
        section("Introduction", "It introduces a model.", index=1),
        section("References", "1. Someone. 2020.", index=2),
    ]
    monkeypatch.setattr(
        summarizer,
        "screening_sections_for_pdf",
        lambda pdf_path, llm_sectioning_prefetch=None: sections,
    )
    calls = []

    def fake_answer(messages):
        calls.append(messages)
        return "First paragraph.\n\nSecond paragraph."

    progress = []
    result = summarizer.summarize_pdf_for_screening(
        Path("paper.pdf"),
        depth=2,
        answer_function=fake_answer,
        progress_callback=progress.append,
    )

    assert result.text == "First paragraph.\n\nSecond paragraph."
    assert [note.section_title for note in result.section_notes] == [
        "Abstract",
        "Introduction",
    ]
    assert progress == [
        "Selecting sections for screening summary...",
        "Summarizing selected sections: Abstract, Introduction...",
    ]
    assert len(calls) == 1
    user_prompt = calls[0][-1]["content"]
    assert "## Abstract" in user_prompt
    assert "## Introduction" in user_prompt
    assert "References" not in user_prompt
    assert "Aim for about 2 short, modular prose paragraph" in user_prompt
    assert "Avoid one oversized run-on paragraph" in user_prompt
    assert "Use readable plain text" in user_prompt
    assert "Do not use Markdown headings, bold text" in user_prompt
    assert "A concise numbered or bulleted list is acceptable only when" in user_prompt
    assert "Do not include meta-commentary" in user_prompt
    assert "researchers interested in" in user_prompt
    assert "Major Point 1" in user_prompt


def test_summarize_pdf_for_screening_is_genre_agnostic(monkeypatch):
    sections = [section("A Strange Essay", "This document argues about ideas.")]
    monkeypatch.setattr(
        summarizer,
        "screening_sections_for_pdf",
        lambda pdf_path, llm_sectioning_prefetch=None: sections,
    )
    prompts = []

    def fake_answer(messages):
        prompts.append("\n".join(message["content"] for message in messages))
        return "summary"

    summarizer.summarize_pdf_for_screening(
        Path("essay.pdf"),
        answer_function=fake_answer,
    )

    combined = "\n".join(prompts)
    assert "genre-agnostic" in combined
    assert "do not assume this is an empirical paper" in combined.lower()
    assert "do not invent" in combined.lower()
    assert "Do not mention Kurrent" in combined
    assert "Do not make claims about whether it is relevant to Kurrent" in combined
    assert "Focus on what the document appears to be about" in combined
