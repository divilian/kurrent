from pathlib import Path
import json
from types import SimpleNamespace

from kurrent.converser import (
    ConverseEngine,
    ConversationState,
    ConverseTurn,
    EvidencePacket,
    build_evidence_packets,
    build_research_inquiry_messages,
    build_retrieval_query,
    call_ollama_chat,
    citation_for_hit,
    evidence_sources,
    format_evidence_packets,
    source_label_for_hit,
    user_facing_pdf_name,
)
from kurrent.schema import ChunkHit


def make_hit(
    chunk_id="doc-1:section-aware-fixed-char-2000-v2:0",
    text="Homophily affects tie formation in adaptive networks.",
    title="Adaptive Networks Paper",
    page_start=3,
    page_end=4,
    section_number="2",
    section_title="Model",
    distance=0.1234,
):
    return ChunkHit(
        chunk_id=chunk_id,
        distance=distance,
        text=text,
        path=Path("/tmp/adaptive-networks.pdf"),
        title=title,
        page_start=page_start,
        page_end=page_end,
        section_number=section_number,
        section_title=section_title,
    )


def test_build_retrieval_query_first_turn_preserves_user_wording():
    """First-turn retrieval should not scholasticize or expand the query."""

    query = build_retrieval_query(
        " using homophily as a basis for dissolving ties in network-based ABMs "
    )

    assert query == "using homophily as a basis for dissolving ties in network-based ABMs"


def test_build_retrieval_query_followup_adds_conversation_focus_not_synonyms():
    """Follow-up retrieval includes recent focus so short refinements make sense."""

    state = ConversationState()
    state.add_turn(
        ConverseTurn(
            user_text="homophily-based tie dissolution in ABMs",
            retrieval_query="homophily-based tie dissolution in ABMs",
            assistant_text="stub",
            evidence=(),
        )
    )

    query = build_retrieval_query("specifically network rewiring", state)

    assert "Recent conversation focus" in query
    assert "homophily-based tie dissolution in ABMs" in query
    assert "Current user question or refinement" in query
    assert "specifically network rewiring" in query


def test_build_evidence_packets_preserves_source_page_section_and_chunk_provenance():
    """Semantic hits become user-facing page evidence plus hidden chunk provenance."""

    packets = build_evidence_packets([make_hit()])

    assert packets == (
        EvidencePacket(
            evidence_id=1,
            chunk_id="doc-1:section-aware-fixed-char-2000-v2:0",
            citation="adaptive-networks.pdf, pp. 3–4",
            title="Adaptive Networks Paper",
            source_name="adaptive-networks.pdf",
            pages="pp. 3–4",
            section="2 Model",
            distance=0.1234,
            text="Homophily affects tie formation in adaptive networks.",
            source_label="adaptive-networks.pdf",
            pdf_path=Path("/tmp/adaptive-networks.pdf"),
            page_start=3,
            page_end=4,
        ),
    )


def test_format_evidence_packets_sends_only_citation_and_excerpt_to_ollama():
    """Ollama-facing evidence should omit internal provenance/debug fields."""

    packets = build_evidence_packets([make_hit()])
    formatted = format_evidence_packets(packets)

    assert '"citation": "adaptive-networks.pdf, pp. 3–4"' in formatted
    assert '"excerpt": "Homophily affects tie formation in adaptive networks."' in formatted
    assert "Evidence 1" not in formatted
    assert "chunk_id" not in formatted
    assert "distance" not in formatted
    assert "section" not in formatted


def test_user_facing_pdf_name_removes_managed_store_hash_suffix():
    """Fallback PDF citations should hide kurrent managed-store hash suffixes."""

    assert (
        user_facing_pdf_name("papachristou2025--12659d50cfa1.pdf")
        == "papachristou2025.pdf"
    )
    assert user_facing_pdf_name("plain-name.pdf") == "plain-name.pdf"


def test_citation_prefers_author_year_when_hit_provides_it():
    """Author/year metadata wins when available on the semantic hit object."""

    hit = SimpleNamespace(
        authors="Davies and Muller",
        year=2011,
        path=Path("/tmp/davies2011--abcdef123456.pdf"),
        title="Underwater Basket-weaving Basics",
        page_start=5,
        page_end=5,
    )

    assert source_label_for_hit(hit) == "Davies and Muller 2011"
    assert citation_for_hit(hit) == "Davies and Muller 2011, p. 5"


def test_citation_can_use_parent_document_metadata_from_sqlite():
    """Converser can cite author/year from parent documents, not only ChunkHit."""

    hit = SimpleNamespace(
        path=Path("/tmp/davies2011--abcdef123456.pdf"),
        title="Underwater Basket-weaving Basics",
        page_start=5,
        page_end=5,
    )
    document = SimpleNamespace(
        authors="Davies and Muller",
        year=2011,
        title="Underwater Basket-weaving Basics",
    )

    assert source_label_for_hit(hit, document=document) == "Davies and Muller 2011"
    assert citation_for_hit(hit, document=document) == "Davies and Muller 2011, p. 5"


def test_citation_falls_back_to_clean_pdf_before_title():
    """When author/year are missing, prefer cleaned filename before title."""

    hit = SimpleNamespace(
        path=Path("/tmp/papachristou2025--12659d50cfa1.pdf"),
        title="Network formation and dynamics among multi-LLMs",
        page_start=7,
        page_end=8,
    )

    assert citation_for_hit(hit) == "papachristou2025.pdf, pp. 7–8"


def test_build_evidence_packets_uses_document_lookup_for_citation():
    """Evidence citation should use parent-document author/year when supplied."""

    document = SimpleNamespace(authors="Davies and Muller", year=2011)
    packets = build_evidence_packets(
        [make_hit(page_start=5, page_end=5)],
        document_lookup=lambda hit: document,
    )

    assert packets[0].citation == "Davies and Muller 2011, p. 5"


def test_evidence_sources_group_chunks_by_pdf_and_keep_first_page():
    """Source navigation should group chunks by PDF and open at the first page."""

    packets = build_evidence_packets(
        [
            make_hit(
                chunk_id="doc-1:section-aware-fixed-char-2000-v2:0",
                page_start=3,
                page_end=4,
            ),
            make_hit(
                chunk_id="doc-1:section-aware-fixed-char-2000-v2:1",
                page_start=7,
                page_end=8,
            ),
        ]
    )

    sources = evidence_sources(packets)

    assert len(sources) == 1
    assert sources[0].source_number == 1
    assert sources[0].citation == "adaptive-networks.pdf, pp. 3–4; pp. 7–8"
    assert sources[0].pdf_path == Path("/tmp/adaptive-networks.pdf")
    assert sources[0].page_start == 3
    assert sources[0].evidence_count == 2


def test_build_research_inquiry_messages_frames_task_as_literature_assessment():
    """The Ollama prompt should ask for RQ assessment, not ordinary QA."""

    messages = build_research_inquiry_messages(
        user_text="Has homophily-triggered tie dissolution been studied?",
        evidence=build_evidence_packets([make_hit()]),
    )

    joined = "\n".join(message["content"] for message in messages)

    assert "Research Inquiry assistant" in joined
    assert "Closest Kurrent evidence" in joined
    assert "Possible novelty angle" in joined
    assert "Coverage assessment" not in joined
    assert "Cautions / next steps" not in joined
    assert "search terms" not in joined
    assert "Use only the provided JSON evidence array" in joined
    assert "citation field as the exact source phrase" in joined


def test_converse_engine_retrieves_once_and_records_state():
    """One converse turn should retrieve with Chroma-facing searcher and store state."""

    class FakeStateStore:
        def get_document(self, doc_id):
            return SimpleNamespace(authors="Davies and Muller", year=2011)

    class FakeSearcher:
        def __init__(self):
            self.calls = []
            self.state_store = FakeStateStore()

        def semantic_chunk_search(
            self,
            search_text,
            n_results=10,
            max_distance=None,
            include_reference_sections=False,
        ):
            self.calls.append(
                {
                    "search_text": search_text,
                    "n_results": n_results,
                    "max_distance": max_distance,
                    "include_reference_sections": include_reference_sections,
                }
            )
            return [make_hit()]

    captured = {}

    def fake_answer(messages):
        captured["messages"] = messages
        return "Corpus-scoped answer."

    searcher = FakeSearcher()
    engine = ConverseEngine(
        searcher=searcher,
        top_k=5,
        max_distance=0.9,
        include_reference_sections=True,
        answer_function=fake_answer,
    )

    turn = engine.answer_user_turn("homophily-based tie dissolution")

    assert searcher.calls == [
        {
            "search_text": "homophily-based tie dissolution",
            "n_results": 5,
            "max_distance": 0.9,
            "include_reference_sections": True,
        }
    ]
    assert turn.assistant_text == "Corpus-scoped answer."
    assert len(turn.evidence) == 1
    assert turn.evidence[0].citation == "Davies and Muller 2011, pp. 3–4"
    assert engine.state.turns == [turn]
    assert "homophily-based tie dissolution" in captured["messages"][1]["content"]
    assert "Evidence 1" not in captured["messages"][1]["content"]
    assert "chunk_id" not in captured["messages"][1]["content"]


def test_converse_engine_second_turn_uses_recent_focus_for_retrieval():
    """Second-turn retrieval should be stateful without LLM query expansion."""

    class FakeSearcher:
        def __init__(self):
            self.queries = []

        def semantic_chunk_search(self, search_text, **kwargs):
            self.queries.append(search_text)
            return [make_hit()]

    searcher = FakeSearcher()
    engine = ConverseEngine(
        searcher=searcher,
        answer_function=lambda messages: "answer",
    )

    engine.answer_user_turn("homophily-based tie dissolution")
    engine.answer_user_turn("no, specifically when dissimilarity causes cutting ties")

    assert searcher.queries[0] == "homophily-based tie dissolution"
    assert "Recent conversation focus" in searcher.queries[1]
    assert "homophily-based tie dissolution" in searcher.queries[1]
    assert "dissimilarity causes cutting ties" in searcher.queries[1]


def test_converse_engine_reports_progress_between_slow_stages():
    """CLI callers can show finer-grained status instead of one vague message."""

    class FakeSearcher:
        def semantic_chunk_search(self, search_text, **kwargs):
            return [make_hit()]

    progress_messages = []
    engine = ConverseEngine(
        searcher=FakeSearcher(),
        answer_function=lambda messages: "answer",
    )

    engine.answer_user_turn(
        "adaptive networks and cooperation",
        progress_callback=progress_messages.append,
    )

    assert progress_messages == [
        "Preparing retrieval query...",
        "Searching Kurrent semantic index...",
        "Retrieved 1 candidate chunk.",
        "Building evidence packet for Ollama...",
        "Asking Ollama for a corpus-grounded assessment...",
        "Recording this turn in the conversation state...",
    ]


def test_call_ollama_chat_streams_tokens_and_returns_full_answer(monkeypatch):
    """Streaming Ollama calls should expose live tokens and keep final text."""

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([
                b'{"message":{"content":"Closest"},"done":false}\n',
                b'{"message":{"content":" evidence"},"done":false}\n',
                b'{"done":true}\n',
            ])

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("kurrent.converser.urlopen", fake_urlopen)

    tokens = []
    answer = call_ollama_chat(
        [{"role": "user", "content": "hello"}],
        model="test-model",
        ollama_url="http://ollama.example",
        timeout_seconds=12.5,
        token_callback=tokens.append,
    )

    assert tokens == ["Closest", " evidence"]
    assert answer == "Closest evidence"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["model"] == "test-model"
    assert captured["timeout"] == 12.5
