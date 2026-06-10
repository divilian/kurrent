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
