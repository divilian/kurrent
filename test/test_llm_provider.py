import json

import pytest

from kurrent.llm_provider import (
    LLMCallError,
    LLMConfigurationError,
    call_llm_chat,
    get_openai_api_key,
    resolve_llm_model_spec,
    validate_llm_model_spec,
)


class FakeResponse:
    def __init__(self, payload=None, lines=None):
        self.payload = payload
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        return iter(self.lines)


def test_explicit_ollama_colon_spec_selects_ollama():
    spec = resolve_llm_model_spec(
        "ollama:llama3.1:kurrent-relevance-8b-instruct-q4_K_M"
    )

    assert spec.provider == "ollama"
    assert spec.model == "llama3.1:kurrent-relevance-8b-instruct-q4_K_M"


def test_bare_model_spec_is_configuration_error():
    with pytest.raises(LLMConfigurationError):
        resolve_llm_model_spec("llama3.1:kurrent-relevance-8b-instruct-q4_K_M")


def test_openai_colon_spec_requires_configured_key(monkeypatch):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "kurrent.llm_provider.load_dotenv",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(LLMConfigurationError):
        validate_llm_model_spec("openai:gpt-4.1-mini")


def test_openai_key_can_come_from_expected_env_name(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert get_openai_api_key() == "sk-test"


def test_openai_chat_call_posts_chat_completion_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": "  useful answer  "}},
                ],
            }
        )

    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setattr("kurrent.llm_provider.urlopen", fake_urlopen)

    content = call_llm_chat(
        [{"role": "user", "content": "hello"}],
        model="openai:gpt-4.1-mini",
        timeout_seconds=12.0,
        temperature=0.0,
        json_format=True,
        num_predict=100,
    )

    assert content == "useful answer"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["timeout"] == 12.0
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gpt-4.1-mini"
    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "hello"},
    ]
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["max_completion_tokens"] == 100


def test_openai_chat_call_streams_tokens(monkeypatch):
    tokens = []

    def fake_urlopen(request, timeout):
        return FakeResponse(
            lines=[
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n',
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
                b"data: [DONE]\n",
            ],
        )

    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setattr("kurrent.llm_provider.urlopen", fake_urlopen)

    content = call_llm_chat(
        [{"role": "user", "content": "hello"}],
        model="openai://gpt-4.1-mini",
        timeout_seconds=12.0,
        stream=True,
        token_callback=tokens.append,
    )

    assert content == "hello"
    assert tokens == ["hel", "lo"]


def test_openai_url_spec_extracts_model_from_query():
    spec = resolve_llm_model_spec(
        "https://api.openai.com/v1/chat/completions?model=gpt-4.1-mini"
    )

    assert spec.provider == "openai"
    assert spec.model == "gpt-4.1-mini"
    assert spec.endpoint_url == "https://api.openai.com/v1/chat/completions"


def test_unsupported_remote_url_is_configuration_error():
    with pytest.raises(LLMConfigurationError):
        resolve_llm_model_spec("https://example.com/v1/chat/completions?model=x")


def _http_error(status_code, body, *, reason="Error", headers=None):
    import io
    from urllib.error import HTTPError

    return HTTPError(
        "https://api.openai.com/v1/chat/completions",
        status_code,
        reason,
        headers or {},
        io.BytesIO(body.encode("utf-8")),
    )


def test_openai_transient_http_error_retries(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if len(calls) < 3:
            raise _http_error(
                500,
                '{"error":{"message":"temporary server trouble","type":"server_error"}}',
                reason="Internal Server Error",
            )
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": "  recovered answer  "}},
                ],
            }
        )

    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setattr("kurrent.llm_provider.urlopen", fake_urlopen)
    monkeypatch.setattr("kurrent.llm_provider.time.sleep", lambda _seconds: None)

    content = call_llm_chat(
        [{"role": "user", "content": "hello"}],
        model="openai:gpt-4.1-mini",
        max_retries=2,
    )

    assert content == "recovered answer"
    assert len(calls) == 3


def test_openai_auth_error_does_not_retry_and_reports_cause(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise _http_error(
            401,
            '{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}',
            reason="Unauthorized",
            headers={"x-request-id": "req_test_123"},
        )

    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setattr("kurrent.llm_provider.urlopen", fake_urlopen)
    monkeypatch.setattr("kurrent.llm_provider.time.sleep", lambda _seconds: None)

    with pytest.raises(LLMCallError) as excinfo:
        call_llm_chat(
            [{"role": "user", "content": "hello"}],
            model="openai:gpt-4.1-mini",
            max_retries=2,
        )

    message = str(excinfo.value)
    assert len(calls) == 1
    assert excinfo.value.retryable is False
    assert excinfo.value.status_code == 401
    assert "OpenAI HTTP 401" in message
    assert "Incorrect API key provided" in message
    assert "request_id=req_test_123" in message


def test_explicit_ollama_chat_uses_local_ollama_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "  local answer  "}})

    monkeypatch.setattr("kurrent.llm_provider.urlopen", fake_urlopen)

    content = call_llm_chat(
        [{"role": "user", "content": "hello"}],
        model="ollama:qwen3:kurrent-rag-8b",
        max_retries=0,
    )

    assert content == "local answer"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["model"] == "qwen3:kurrent-rag-8b"
