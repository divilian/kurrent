import pytest

from kurrent.llm_sectioner import (
    LLMSectioningUnavailableError,
    OllamaTimeoutError,
    select_section_headings_with_ollama,
)
from kurrent.sectioner import HeadingCandidate


def make_candidate(candidate_id: int, text: str) -> HeadingCandidate:
    return HeadingCandidate(
        candidate_id=candidate_id,
        line_index=candidate_id,
        page=candidate_id + 1,
        line_text=text,
        previous_lines=[],
        next_lines=["Body text."],
        features={},
        candidate_text=text,
    )


def test_section_llm_stops_after_repeated_candidate_failures(monkeypatch):
    candidates = [
        make_candidate(0, "I. INTRODUCTION"),
        make_candidate(1, "II. MODEL"),
        make_candidate(2, "III. RESULTS"),
    ]

    calls = {"count": 0}

    def always_timeout(**_kwargs):
        calls["count"] += 1
        raise OllamaTimeoutError("timeout")

    monkeypatch.setattr("kurrent.llm_sectioner._ollama_chat", always_timeout)

    with pytest.raises(LLMSectioningUnavailableError):
        select_section_headings_with_ollama(
            candidates,
            max_consecutive_failures=2,
            timeout_seconds=1,
            singleton_timeout_seconds=1,
        )

    # Each failed candidate tries JSON twice plus the YES/NO fallback once.
    # After two failed candidates, the third should not be attempted.
    assert calls["count"] == 6
