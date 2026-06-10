import os
import sys

from kurrent.embedder import suppress_tqdm_for_model_loading


def test_suppress_tqdm_for_model_loading_silences_stdout_and_stderr(
    capsys,
    monkeypatch,
):
    monkeypatch.delenv("KURRENT_SHOW_MODEL_LOAD_PROGRESS", raising=False)
    monkeypatch.delenv("TQDM_DISABLE", raising=False)

    with suppress_tqdm_for_model_loading():
        assert os.environ["TQDM_DISABLE"] == "1"
        print("Loading weights: 100%")
        print("Loading weights: 100%", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "TQDM_DISABLE" not in os.environ


def test_suppress_tqdm_for_model_loading_restores_existing_tqdm_setting(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "0")
    monkeypatch.delenv("KURRENT_SHOW_MODEL_LOAD_PROGRESS", raising=False)

    with suppress_tqdm_for_model_loading():
        assert os.environ["TQDM_DISABLE"] == "1"

    assert os.environ["TQDM_DISABLE"] == "0"


def test_suppress_tqdm_for_model_loading_can_show_progress_for_debugging(
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("KURRENT_SHOW_MODEL_LOAD_PROGRESS", "1")

    with suppress_tqdm_for_model_loading():
        print("visible model-load output")

    captured = capsys.readouterr()
    assert "visible model-load output" in captured.out


def test_query_chunks_uses_embedder_model_for_query_embedding():
    """Search queries should use the same SentenceTransformer as indexed chunks."""
    import numpy as np

    from kurrent.embedder import Embedder

    class FakeModel:
        def __init__(self):
            self.encoded_texts = []

        def encode(self, texts, convert_to_numpy=True):
            self.encoded_texts.append(list(texts))
            assert convert_to_numpy is True
            return np.array([[0.1, 0.2, 0.3]])

    class FakeCollection:
        def __init__(self):
            self.query_kwargs = None

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return {
                "ids": [["doc-1:chunker:0"]],
                "documents": [["matching text"]],
                "metadatas": [[{"doc_id": "doc-1"}]],
                "distances": [[0.123]],
            }

    embedder = Embedder.__new__(Embedder)
    embedder.model = FakeModel()
    embedder.collection = FakeCollection()

    matches = embedder.query_chunks("personal knowledge base", n_results=3)

    assert embedder.model.encoded_texts == [["personal knowledge base"]]
    assert embedder.collection.query_kwargs == {
        "query_embeddings": [[0.1, 0.2, 0.3]],
        "n_results": 3,
        "include": ["documents", "metadatas", "distances"],
    }
    assert matches[0].chunk_id == "doc-1:chunker:0"
    assert matches[0].distance == 0.123
    assert matches[0].text == "matching text"
