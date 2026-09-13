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
        "Summarizing selected sections: Abstract and Introduction...",
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


def test_screening_summary_progress_uses_and_for_final_section(monkeypatch):
    """Verify selected-section status reads like a complete English list."""

    sections = [
        section("front matter", "Opening material.", index=0),
        section("Introduction", "Background.", index=1),
        section("Methods", "Methods.", index=2),
        section("Results", "Results.", index=3),
    ]
    monkeypatch.setattr(
        summarizer,
        "screening_sections_for_pdf",
        lambda pdf_path, llm_sectioning_prefetch=None: sections,
    )

    progress = []
    summarizer.summarize_pdf_for_screening(
        Path("paper.pdf"),
        depth=2,
        answer_function=lambda _messages: "Summary.",
        progress_callback=progress.append,
    )

    assert progress[-1] == (
        "Summarizing selected sections: front matter, Introduction, Methods, and Results..."
    )


def test_deliberate_summary_uses_all_substantive_chunks_in_one_call():
    from test.factories import make_chunk

    chunks = [
        make_chunk(
            "doc",
            0,
            "The paper introduces a network model.",
            section_title="Introduction",
        ),
        make_chunk(
            "doc",
            1,
            "The model produces a cooperation result.",
            section_title="Results",
        ),
        make_chunk(
            "doc",
            2,
            "1. Someone. 2020.",
            section_title="References",
        ),
    ]
    calls = []

    def fake_answer(messages):
        calls.append(messages)
        return "A complete summary."

    result = summarizer.summarize_document_chunks(
        chunks,
        title="A Paper",
        authors="A. Author",
        year=2026,
        answer_function=fake_answer,
    )

    assert result.text == "A complete summary."
    assert result.source_chunk_count == 2
    assert len(calls) == 1
    prompt = calls[0][-1]["content"]
    assert "The paper introduces a network model." in prompt
    assert "The model produces a cooperation result." in prompt
    assert "Someone. 2020" not in prompt
    assert "Title: A Paper" in prompt
    assert "This is a deliberate post-ingest summary" in prompt


def test_deliberate_summary_batches_long_documents_before_synthesis():
    from test.factories import make_chunk

    chunks = [
        make_chunk(
            "doc",
            0,
            "A" * 3000,
            section_title="First",
        ),
        make_chunk(
            "doc",
            1,
            "B" * 3000,
            section_title="Second",
        ),
    ]
    calls = []
    progress = []

    def fake_answer(messages):
        calls.append(messages)
        prompt = messages[-1]["content"]
        if "<document portion>" in prompt:
            if "AAAA" in prompt:
                return "Notes from first part."
            return "Notes from second part."
        return "Final synthesized summary."

    result = summarizer.summarize_document_chunks(
        chunks,
        max_num_ctx=3500,
        progress_callback=progress.append,
        answer_function=fake_answer,
    )

    assert result.text == "Final synthesized summary."
    assert result.source_chunk_count == 2
    assert len(calls) == 3
    assert progress == [
        "Summarizing document part 1 of 2...",
        "Summarizing document part 2 of 2...",
        "Writing complete summary...",
    ]
    final_prompt = calls[-1][-1]["content"]
    assert "Notes from first part." in final_prompt
    assert "Notes from second part." in final_prompt
    assert "detailed notes" in final_prompt


def test_deliberate_openai_summary_retries_output_limit_with_larger_budget(
    monkeypatch,
):
    calls = []

    def fake_call_llm_chat(messages, **kwargs):
        calls.append(kwargs["num_predict"])
        if len(calls) == 1:
            raise summarizer.LLMCallError(
                "OpenAI response incomplete for gpt-5.6-luna: "
                "max_output_tokens",
                provider="OpenAI",
                model="gpt-5.6-luna",
            )
        return "Detailed notes."

    monkeypatch.setattr(summarizer, "call_llm_chat", fake_call_llm_chat)

    result = summarizer._call_document_summary_chat(
        [{"role": "user", "content": "Summarize this part."}],
        model="openai:gpt-5.6-luna",
        timeout_seconds=120.0,
        num_ctx=32_768,
        num_predict=1_000,
        answer_function=None,
    )

    assert result == "Detailed notes."
    assert calls == [4_000, 8_000]


def test_deliberate_summary_reports_note_condensation_calls():
    from test.factories import make_chunk

    chunks = [
        make_chunk("doc", 0, "A" * 3000, section_title="First"),
        make_chunk("doc", 1, "B" * 3000, section_title="Second"),
    ]
    progress = []
    call_count = 0

    def fake_answer(messages):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return "N" * 3000
        if call_count <= 4:
            return "Condensed notes."
        return "Final summary."

    result = summarizer.summarize_document_chunks(
        chunks,
        max_num_ctx=3500,
        progress_callback=progress.append,
        answer_function=fake_answer,
    )

    assert result.text == "Final summary."
    assert progress == [
        "Summarizing document part 1 of 2...",
        "Summarizing document part 2 of 2...",
        "Condensing summary notes part 1 of 2...",
        "Condensing summary notes part 2 of 2...",
        "Writing complete summary...",
    ]


def test_deliberate_ollama_summary_keeps_existing_output_budget(monkeypatch):
    calls = []

    def fake_call_llm_chat(messages, **kwargs):
        calls.append(kwargs["num_predict"])
        return "Detailed notes."

    monkeypatch.setattr(summarizer, "call_llm_chat", fake_call_llm_chat)

    result = summarizer._call_document_summary_chat(
        [{"role": "user", "content": "Summarize this part."}],
        model="ollama:qwen2.5:kurrent-summarization-7b",
        timeout_seconds=120.0,
        num_ctx=32_768,
        num_predict=1_000,
        answer_function=None,
    )

    assert result == "Detailed notes."
    assert calls == [1_000]
