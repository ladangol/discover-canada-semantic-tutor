"""The ONLY module that knows Venice exists.

    generate(messages, model_id=None) -> GenerationResult

The API key is read from the environment at call time, used only as the
client credential, and never logged or included in error messages.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Iterator
from functools import lru_cache
import os

from openai import OpenAI, OpenAIError

from src import budget
from src.config import (
    NOT_CONFIGURED_MESSAGE,
    generation_configured,
    generation_model,
    venice_base_url,
)

DEFAULT_MAX_TOKENS = 1500
# Venice answers 429 "model is currently overloaded" when a model is at capacity, which is
# unrelated to account credits. Retry a few times with backoff before giving up.
log = logging.getLogger("discover_canada.generation")
RETRY_DELAYS = (2.0, 5.0, 10.0)
REQUEST_TIMEOUT_S = 120.0     # a hung request must fail and retry, not block a user for the SDK's 10 minutes
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


class GenerationUnavailable(RuntimeError):
    """The AI is not available right now. `reason` says why; str(exc) is safe to show to a learner."""

    MESSAGES = {
        "budget": "The study helper has used up today’s AI allowance. Anything already created is still here, "
                  "and new questions and summaries are back tomorrow.",
        "busy": "The AI is busy right now. Please try again in a minute.",
        "config": "The study tools are temporarily unavailable.",
        "policy": "I can’t help with that one. Try rephrasing your question.",
        "library": "That one isn’t in the demo’s ready-made library yet. Try a chapter or one of the examples.",
    }

    def __init__(self, reason: str, message: str | None = None):
        self.reason = reason
        super().__init__(message or self.MESSAGES[reason])


class GenerationNotConfigured(GenerationUnavailable):
    """Raised when VENICE_API_KEY is missing. Retrieval still works."""

    def __init__(self, message: str = NOT_CONFIGURED_MESSAGE):
        super().__init__("config", message)


def classify_error(exc: Exception) -> tuple[str, bool]:
    """Map a provider failure to (reason, retryable). Duck-typed on status_code/body so it never depends on
    the SDK's exact exception classes, and it never inspects or logs request content."""
    status = getattr(exc, "status_code", None)
    blob = str(getattr(exc, "body", "") or "")
    try:
        blob += getattr(getattr(exc, "response", None), "text", "") or ""
    except Exception:                                   # response body may be unreadable
        pass
    blob = blob.upper()
    if status == 402 or "INSUFFICIENT_BALANCE" in blob or "SPEND_LIMIT_EXCEEDED" in blob:
        return "budget", False                          # balance / key spend limit: retrying only wastes calls
    if status in (401, 403):
        return "config", False
    if status == 422 and "CONTENT_POLICY" in blob:
        return "policy", False
    return "busy", True                                 # 429 overload/rate limit, 5xx, timeouts, connection errors


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not generation_configured():
        raise GenerationNotConfigured(NOT_CONFIGURED_MESSAGE)
    return OpenAI(api_key=os.environ["VENICE_API_KEY"], base_url=venice_base_url(), timeout=REQUEST_TIMEOUT_S)


def generate(
    messages: list[dict[str, str]],
    model_id: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    client = _client()                       # raises GenerationNotConfigured first when there is no key
    try:
        budget.check()
    except budget.DailyBudgetExceeded:
        raise GenerationUnavailable("budget") from None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = client.chat.completions.create(
                model=model_id or generation_model(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"venice_parameters": {"include_venice_system_prompt": False}},
            )
            break
        except OpenAIError as exc:
            # Deliberately drop the SDK message (it can echo request details); log only status + type.
            reason, retryable = classify_error(exc)
            log.warning("Venice call failed: status=%s type=%s reason=%s attempt=%d",
                        getattr(exc, "status_code", None), type(exc).__name__, reason, attempt + 1)
            if reason == "budget":
                budget.trip()                       # stop calling the provider until tomorrow
            if retryable and attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise GenerationUnavailable(reason) from None

    choice = response.choices[0]
    content = choice.message.content
    text = _THINK_BLOCK.sub("", content).strip() if isinstance(content, str) else ""
    if not text:
        reason = getattr(choice, "finish_reason", None)
        used = getattr(getattr(response, "usage", None), "completion_tokens", None)
        raise RuntimeError(f"Venice returned an empty text response (finish_reason={reason}, "
                           f"completion_tokens={used}, max_tokens={max_tokens}).")

    usage = getattr(response, "usage", None)
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    budget.record(pt if pt is not None else sum(len(m["content"]) for m in messages) // 4,
                  ct if ct is not None else len(text) // 4)
    return GenerationResult(
        text=text,
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )



# --- streaming (Milestone 4 follow-up: speak while the answer is still being written) -------------------------------------------------------

class _ThinkFilter:
    """Removes <think>…</think> from a text stream, even when a tag is split across two chunks (the non-streaming path strips it with a regex)."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf, self._inside = "", False

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-n:].lower()):
                return n
        return 0

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            low = self._buf.lower()
            if self._inside:
                i = low.find(self.CLOSE)
                if i < 0:
                    self._buf = self._buf[len(self._buf) - self._partial_suffix(self._buf, self.CLOSE):]
                    break
                self._buf, self._inside = self._buf[i + len(self.CLOSE):], False
            else:
                i = low.find(self.OPEN)
                if i < 0:
                    keep = self._partial_suffix(self._buf, self.OPEN)
                    out.append(self._buf[: len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep:]
                    break
                out.append(self._buf[:i])
                self._buf, self._inside = self._buf[i + len(self.OPEN):], True
        return "".join(out)

    def flush(self) -> str:
        rest, self._buf = ("" if self._inside else self._buf), ""
        return rest


class GenerationStream:
    """The answer as it is written: iterate for text pieces; `.result` (a GenerationResult, like `generate` returns) is set when the stream ends.

    Same guarantees as `generate`: key check, daily budget, retries with backoff, learner-safe `GenerationUnavailable`. A failure BEFORE the first
    text is retried; one after text has been delivered is raised (the caller already has part of an answer)."""

    def __init__(self, messages: list[dict[str, str]], model_id: str | None, temperature: float, max_tokens: int):
        self._messages, self._model, self._temperature, self._max_tokens = messages, model_id, temperature, max_tokens
        self.result: GenerationResult | None = None

    def _open(self):
        client = _client()
        try:
            budget.check()
        except budget.DailyBudgetExceeded:
            raise GenerationUnavailable("budget") from None
        return client

    def __iter__(self) -> Iterator[str]:
        client = self._open()
        for attempt in range(len(RETRY_DELAYS) + 1):
            delivered, finish, usage, parts = False, None, None, []
            flt = _ThinkFilter()
            try:
                stream = client.chat.completions.create(
                    model=self._model or generation_model(), messages=self._messages, temperature=self._temperature,
                    max_tokens=self._max_tokens, stream=True,
                    extra_body={"venice_parameters": {"include_venice_system_prompt": False}})
                for chunk in stream:
                    usage = getattr(chunk, "usage", None) or usage
                    if not getattr(chunk, "choices", None):
                        continue
                    ch = chunk.choices[0]
                    finish = getattr(ch, "finish_reason", None) or finish
                    piece = flt.feed(getattr(ch.delta, "content", None) or "")
                    if not delivered:
                        piece = piece.lstrip()
                    if piece:
                        delivered = True
                        parts.append(piece)
                        yield piece
                tail = flt.flush() if delivered else flt.flush().lstrip()
                if tail:
                    delivered = True
                    parts.append(tail)
                    yield tail
                break
            except OpenAIError as exc:
                reason, retryable = classify_error(exc)
                log.warning("Venice stream failed: status=%s type=%s reason=%s attempt=%d delivered=%s",
                            getattr(exc, "status_code", None), type(exc).__name__, reason, attempt + 1, delivered)
                if reason == "budget":
                    budget.trip()
                if retryable and not delivered and attempt < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                raise GenerationUnavailable(reason) from None
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError(f"Venice returned an empty text response (finish_reason={finish}, max_tokens={self._max_tokens}).")
        pt, ct = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        budget.record(pt if pt is not None else sum(len(m["content"]) for m in self._messages) // 4, ct if ct is not None else len(text) // 4)
        self.result = GenerationResult(text=text, finish_reason=finish, prompt_tokens=pt, completion_tokens=ct)


def generate_stream(messages: list[dict[str, str]], model_id: str | None = None, *, temperature: float = 0.0,
                    max_tokens: int = DEFAULT_MAX_TOKENS) -> GenerationStream:
    """`generate`, but the answer arrives as it is written: `for piece in generate_stream(...)`, then `.result`."""
    return GenerationStream(messages, model_id, temperature, max_tokens)
