"""Provider-neutral chat calls for Kurrent LLM roles.

Kurrent LLM settings use explicit provider prefixes, such as
``ollama:qwen3:kurrent-rag-8b`` or ``openai:gpt-4.1-mini``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse, urlunparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from kurrent.config import DEFAULT_OLLAMA_URL

__all__ = [
    "DEFAULT_OPENAI_CHAT_URL",
    "LLMCallError",
    "LLMConfigurationError",
    "LLMModelSpec",
    "call_llm_chat",
    "get_openai_api_key",
    "is_ollama_model_spec",
    "llm_provider_label",
    "resolve_llm_model_spec",
    "validate_llm_model_spec",
]

DEFAULT_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_CHAT_PATH_SUFFIX = "/chat/completions"


class LLMConfigurationError(RuntimeError):
    """Raised when a configured LLM model spec cannot be used."""


class LLMCallError(RuntimeError):
    """Raised when a configured LLM provider call fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True, slots=True)
class LLMModelSpec:
    """A parsed Kurrent LLM model specification."""

    provider: str
    model: str
    endpoint_url: str | None = None
    raw: str = ""

    @property
    def label(self) -> str:
        if self.provider == "ollama":
            return "Ollama"
        if self.provider == "openai":
            return "OpenAI"
        return self.provider


def _strip_provider_payload(raw: str, prefix: str) -> str:
    payload = raw[len(prefix):]
    while payload.startswith("/"):
        payload = payload[1:]
    return payload.strip()


def _looks_like_absolute_url(raw: str) -> bool:
    return re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) is not None


def _openai_chat_url_from_env() -> str:
    load_dotenv()
    return (
        os.environ.get("KURRENT_OPENAI_CHAT_URL")
        or os.environ.get("OPENAI_CHAT_URL")
        or DEFAULT_OPENAI_CHAT_URL
    )


def _parse_openai_url(raw: str) -> LLMModelSpec:
    parsed = urlparse(raw)
    host = parsed.netloc.lower()

    if "openai" not in host or not parsed.path.endswith(OPENAI_CHAT_PATH_SUFFIX):
        raise LLMConfigurationError(
            "Unsupported remote LLM URL. Kurrent currently recognizes OpenAI "
            "chat-completions URLs such as "
            "https://api.openai.com/v1/chat/completions?model=gpt-4.1-mini, "
            "or concise specs such as openai:gpt-4.1-mini."
        )

    query = parse_qs(parsed.query)
    model = ""
    if query.get("model"):
        model = query["model"][0].strip()
    elif parsed.fragment:
        model = parsed.fragment.strip()

    if not model:
        raise LLMConfigurationError(
            "OpenAI LLM URL is missing a model. Add ?model=<model-name>, "
            "or use a concise spec such as openai:gpt-4.1-mini."
        )

    endpoint = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            "",
            "",
        )
    )
    return LLMModelSpec(
        provider="openai",
        model=model,
        endpoint_url=endpoint,
        raw=raw,
    )


def resolve_llm_model_spec(model: str | None) -> LLMModelSpec:
    """Parse a Kurrent LLM model setting.

    Supported examples:
        ollama:qwen3:kurrent-rag-8b
        openai:gpt-4.1-mini
        https://api.openai.com/v1/chat/completions?model=gpt-4.1-mini
    """

    raw = "" if model is None else str(model).strip()
    if not raw:
        raise LLMConfigurationError(
            "LLM model setting is empty. Use an explicit provider spec such as "
            "ollama:qwen3:kurrent-rag-8b or openai:gpt-4.1-mini."
        )

    lowered = raw.lower()

    if lowered.startswith("openai://"):
        openai_model = _strip_provider_payload(raw, "openai://")
        if not openai_model:
            raise LLMConfigurationError("OpenAI LLM spec is missing a model name.")
        return LLMModelSpec(
            provider="openai",
            model=openai_model,
            endpoint_url=_openai_chat_url_from_env(),
            raw=raw,
        )

    if lowered.startswith("openai:"):
        openai_model = raw.split(":", 1)[1].strip()
        if not openai_model:
            raise LLMConfigurationError("OpenAI LLM spec is missing a model name.")
        return LLMModelSpec(
            provider="openai",
            model=openai_model,
            endpoint_url=_openai_chat_url_from_env(),
            raw=raw,
        )

    if lowered.startswith("ollama://"):
        ollama_model = _strip_provider_payload(raw, "ollama://")
        if not ollama_model:
            raise LLMConfigurationError("Ollama LLM spec is missing a model name.")
        return LLMModelSpec(provider="ollama", model=ollama_model, raw=raw)

    if lowered.startswith("ollama:"):
        ollama_model = raw.split(":", 1)[1].strip()
        if not ollama_model:
            raise LLMConfigurationError("Ollama LLM spec is missing a model name.")
        return LLMModelSpec(provider="ollama", model=ollama_model, raw=raw)

    if lowered.startswith("http://") or lowered.startswith("https://"):
        return _parse_openai_url(raw)

    if _looks_like_absolute_url(raw):
        raise LLMConfigurationError(
            "Unsupported LLM URL scheme. Use ollama:<model>, openai:<model>, "
            "openai://<model>, or an OpenAI chat-completions URL with "
            "?model=<model>."
        )

    raise LLMConfigurationError(
        "LLM model spec must include an explicit provider. Use ollama:<model> "
        "for local Ollama or openai:<model> for OpenAI."
    )


def get_openai_api_key() -> str:
    """Return the configured OpenAI API key from .env or the environment."""

    load_dotenv()
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise LLMConfigurationError(
            "OpenAI LLM selected, but no API key is configured. Set "
            "OPENAI_KEY in .env or in the environment."
        )
    return api_key.strip()


def is_ollama_model_spec(model: str | None) -> bool:
    """Return True when a model spec resolves to local Ollama."""

    return resolve_llm_model_spec(model).provider == "ollama"


def llm_provider_label(model: str | None) -> str:
    """Return a human-friendly provider label for status messages."""

    return resolve_llm_model_spec(model).label


def validate_llm_model_spec(model: str | None) -> LLMModelSpec:
    """Validate provider configuration without making a network call."""

    spec = resolve_llm_model_spec(model)
    if spec.provider == "openai":
        get_openai_api_key()
    return spec



def _configured_llm_max_retries() -> int:
    raw_value = os.environ.get("KURRENT_LLM_MAX_RETRIES", "2")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 2


def _configured_retry_initial_seconds() -> float:
    raw_value = os.environ.get("KURRENT_LLM_RETRY_INITIAL_SECONDS", "1.0")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 1.0


def _configured_retry_max_seconds() -> float:
    raw_value = os.environ.get("KURRENT_LLM_RETRY_MAX_SECONDS", "8.0")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 8.0


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or 500 <= status_code <= 599


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _response_header(headers: object, name: str) -> str | None:
    if headers is None:
        return None
    try:
        value = headers.get(name)  # type: ignore[attr-defined]
    except Exception:
        value = None
    if value is None:
        try:
            value = headers.get(name.lower())  # type: ignore[attr-defined]
        except Exception:
            value = None
    if value is None:
        return None
    return str(value)


def _extract_error_message_from_body(body: str) -> str:
    text = body.strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]

    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        parts: list[str] = []
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())
        error_type = error.get("type")
        if isinstance(error_type, str) and error_type.strip():
            parts.append(f"type={error_type.strip()}")
        code = error.get("code")
        if isinstance(code, str) and code.strip():
            parts.append(f"code={code.strip()}")
        if parts:
            return "; ".join(parts)

    return text[:500]


def _http_error_message(
    exc: HTTPError,
    *,
    provider_label: str,
    model: str,
) -> tuple[str, bool, float | None]:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""

    detail = _extract_error_message_from_body(body) or str(exc.reason)
    retryable = _is_retryable_http_status(exc.code)
    request_id = _response_header(exc.headers, "x-request-id")
    retry_after = _parse_retry_after_seconds(
        _response_header(exc.headers, "retry-after")
    )

    message = f"{provider_label} HTTP {exc.code} for {model}: {detail}"
    if request_id:
        message += f" (request_id={request_id})"
    return message, retryable, retry_after


def _raise_http_llm_call_error(
    exc: HTTPError,
    *,
    provider_label: str,
    model: str,
) -> None:
    message, retryable, retry_after = _http_error_message(
        exc,
        provider_label=provider_label,
        model=model,
    )
    raise LLMCallError(
        message,
        provider=provider_label,
        model=model,
        retryable=retryable,
        status_code=exc.code,
        retry_after_seconds=retry_after,
    ) from exc


def _raise_transient_llm_call_error(
    exc: BaseException,
    *,
    provider_label: str,
    model: str,
) -> None:
    raise LLMCallError(
        f"{provider_label} call failed for {model}: {type(exc).__name__}: {exc}",
        provider=provider_label,
        model=model,
        retryable=True,
    ) from exc


def _request_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    provider_label: str = "LLM",
    model: str = "",
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        _raise_http_llm_call_error(
            exc,
            provider_label=provider_label,
            model=model,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _raise_transient_llm_call_error(
            exc,
            provider_label=provider_label,
            model=model,
        )
    raise AssertionError("unreachable")


def _call_ollama_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    timeout_seconds: float = 120.0,
    temperature: float,
    stream: bool,
    token_callback: Callable[[str], None] | None,
    json_format: bool,
    num_ctx: int | None,
    num_predict: int | None,
) -> str:
    url = f"{DEFAULT_OLLAMA_URL.rstrip('/')}/api/chat"
    options: dict[str, Any] = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": options,
    }
    if json_format:
        payload["format"] = "json"

    if not stream:
        response_data = _request_json(
            url,
            payload,
            timeout_seconds=timeout_seconds,
            provider_label="Ollama",
            model=model,
        )
        content = response_data.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise LLMCallError(
                f"Ollama returned an empty response for {model}.",
                provider="Ollama",
                model=model,
                retryable=True,
            )
        return content.strip()

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                data = json.loads(raw_line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if isinstance(content, str) and content:
                    parts.append(content)
                    if token_callback is not None:
                        token_callback(content)
                if data.get("done"):
                    break
    except HTTPError as exc:
        _raise_http_llm_call_error(
            exc,
            provider_label="Ollama",
            model=model,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _raise_transient_llm_call_error(
            exc,
            provider_label="Ollama",
            model=model,
        )

    content = "".join(parts)
    if not content.strip():
        raise LLMCallError(
            f"Ollama returned an empty response for {model}.",
            provider="Ollama",
            model=model,
            retryable=True,
        )
    return content.strip()


def _openai_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _call_openai_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    endpoint_url: str,
    timeout_seconds: float,
    temperature: float,
    stream: bool,
    token_callback: Callable[[str], None] | None,
    json_format: bool,
    num_predict: int | None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    }
    if num_predict is not None:
        payload["max_completion_tokens"] = num_predict
    if json_format:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {get_openai_api_key()}",
    }

    if not stream:
        response_data = _request_json(
            endpoint_url,
            payload,
            timeout_seconds=timeout_seconds,
            headers=headers,
            provider_label="OpenAI",
            model=model,
        )
        choices = response_data.get("choices", [])
        if not choices:
            raise LLMCallError(
                f"OpenAI returned no choices for {model}.",
                provider="OpenAI",
                model=model,
                retryable=True,
            )
        message = choices[0].get("message", {})
        content = _openai_message_text(message.get("content"))
        if not content.strip():
            raise LLMCallError(
                f"OpenAI returned an empty response for {model}.",
                provider="OpenAI",
                model=model,
                retryable=True,
            )
        return content.strip()

    request = Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    parts: list[str] = []

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_text = line[len("data:"):].strip()
                if data_text == "[DONE]":
                    break
                data = json.loads(data_text)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if isinstance(content, str) and content:
                    parts.append(content)
                    if token_callback is not None:
                        token_callback(content)
    except HTTPError as exc:
        _raise_http_llm_call_error(
            exc,
            provider_label="OpenAI",
            model=model,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        _raise_transient_llm_call_error(
            exc,
            provider_label="OpenAI",
            model=model,
        )

    content = "".join(parts)
    if not content.strip():
        raise LLMCallError(
            f"OpenAI returned an empty response for {model}.",
            provider="OpenAI",
            model=model,
            retryable=True,
        )
    return content.strip()


def _sleep_before_retry(exc: LLMCallError, attempt_index: int) -> None:
    max_delay = _configured_retry_max_seconds()
    if exc.retry_after_seconds is not None:
        delay = min(exc.retry_after_seconds, max_delay)
    else:
        initial_delay = _configured_retry_initial_seconds()
        delay = min(initial_delay * (2 ** max(0, attempt_index - 1)), max_delay)
    if delay > 0:
        time.sleep(delay)


def _call_with_retries(
    call: Callable[[], str],
    *,
    max_retries: int,
    allow_retries: bool,
) -> str:
    for attempt_index in range(max_retries + 1):
        try:
            return call()
        except LLMCallError as exc:
            if (
                not allow_retries
                or not exc.retryable
                or attempt_index >= max_retries
            ):
                raise
            _sleep_before_retry(exc, attempt_index + 1)
    raise AssertionError("unreachable")


def call_llm_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    timeout_seconds: float = 120.0,
    temperature: float = 0.2,
    stream: bool = False,
    token_callback: Callable[[str], None] | None = None,
    json_format: bool = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    max_retries: int | None = None,
) -> str:
    """Call the provider implied by ``model`` and return assistant text.

    Transient provider failures are retried by default. Configuration errors,
    authentication failures, unknown models, and other non-transient failures
    are raised immediately with their provider-specific cause.
    """

    spec = resolve_llm_model_spec(model)
    retry_count = _configured_llm_max_retries() if max_retries is None else max(0, max_retries)

    # Avoid replaying partial tokens to interactive callers. Non-streaming LLM
    # calls are safe to retry because no partial answer has been emitted.
    allow_retries = not stream

    if spec.provider == "ollama":
        return _call_with_retries(
            lambda: _call_ollama_chat(
                messages,
                model=spec.model,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                stream=stream,
                token_callback=token_callback,
                json_format=json_format,
                num_ctx=num_ctx,
                num_predict=num_predict,
            ),
            max_retries=retry_count,
            allow_retries=allow_retries,
        )

    if spec.provider == "openai":
        return _call_with_retries(
            lambda: _call_openai_chat(
                messages,
                model=spec.model,
                endpoint_url=spec.endpoint_url or _openai_chat_url_from_env(),
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                stream=stream,
                token_callback=token_callback,
                json_format=json_format,
                num_predict=num_predict,
            ),
            max_retries=retry_count,
            allow_retries=allow_retries,
        )

    raise LLMConfigurationError(f"Unsupported LLM provider: {spec.provider!r}")
