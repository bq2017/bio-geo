"""OpenAI-compatible chat-completions client used by DS adjudication."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DSResponse:
    content: str
    endpoint: str
    attempts: int
    latency_seconds: float
    usage: dict[str, Any] | None = None
    reasoning: Any = None
    response_message_keys: tuple[str, ...] = ()
    retry_errors: tuple[dict[str, Any], ...] = ()


class DSRequestError(RuntimeError):
    """Terminal request failure with diagnostics from every HTTP attempt."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        endpoint: str,
        latency_seconds: float,
        retry_errors: Iterable[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.endpoint = endpoint
        self.latency_seconds = latency_seconds
        self.retry_errors = tuple(retry_errors)


class DSClient:
    """Small retrying client for an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        endpoints: Iterable[str],
        model: str,
        *,
        timeout: float = 120,
        retries: int = 3,
        retry_delay: float = 0.25,
        request_interval: float = 0.0,
        enable_thinking: bool | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.endpoints = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
        if not self.endpoints:
            raise ValueError("at least one endpoint is required")
        if retries < 1:
            raise ValueError("retries must be at least 1")
        if request_interval < 0:
            raise ValueError("request_interval must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.request_interval = request_interval
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self._next_endpoint = 0
        self._endpoint_lock = threading.Lock()
        self._request_slot_lock = threading.Lock()
        self._next_request_time = 0.0

    def _wait_for_request_slot(self) -> None:
        if not self.request_interval:
            return
        with self._request_slot_lock:
            now = time.monotonic()
            request_time = max(now, self._next_request_time)
            self._next_request_time = request_time + self.request_interval
        delay = request_time - now
        if delay > 0:
            time.sleep(delay)

    def _defer_retry_slot(self) -> None:
        if not self.request_interval:
            return
        with self._request_slot_lock:
            self._next_request_time = max(
                self._next_request_time,
                time.monotonic() + self.request_interval,
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
    ) -> DSResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        last_error: Exception | None = None
        retry_errors: list[dict[str, Any]] = []
        with self._endpoint_lock:
            starting_index = self._next_endpoint
            self._next_endpoint = (self._next_endpoint + 1) % len(self.endpoints)

        for attempt in range(1, self.retries + 1):
            endpoint_index = (starting_index + attempt - 1) % len(self.endpoints)
            endpoint = self.endpoints[endpoint_index]
            request = Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                self._wait_for_request_slot()
                with urlopen(request, timeout=self.timeout) as response:
                    response_body = json.loads(response.read().decode("utf-8"))
                message = response_body["choices"][0]["message"]
                content = message["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty chat completion content")
                return DSResponse(
                    content=content,
                    endpoint=endpoint,
                    attempts=attempt,
                    latency_seconds=round(time.monotonic() - started, 3),
                    usage=response_body.get("usage"),
                    reasoning=message.get("reasoning", message.get("reasoning_content")),
                    response_message_keys=tuple(message),
                    retry_errors=tuple(retry_errors),
                )
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                retry_errors.append(
                    {
                        "attempt": attempt,
                        "endpoint": endpoint,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt < self.retries:
                    self._defer_retry_slot()
                if attempt < self.retries and self.retry_delay:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay * random.uniform(0.8, 1.2))

        endpoint = retry_errors[-1]["endpoint"] if retry_errors else self.endpoints[0]
        raise DSRequestError(
            f"chat completion failed after {self.retries} attempts: {last_error}",
            attempts=self.retries,
            endpoint=endpoint,
            latency_seconds=round(time.monotonic() - started, 3),
            retry_errors=retry_errors,
        ) from last_error


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse the first JSON object from plain text or a Markdown fence."""
    text = content.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        text = fence.group(1).strip()
    object_start = text.find("{")
    if object_start < 0:
        raise ValueError("response does not contain a JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def append_evidence(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
