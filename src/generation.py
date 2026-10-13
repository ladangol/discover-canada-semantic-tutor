"""The ONLY module that knows Venice exists.

    generate(messages, model_id=None) -> GenerationResult

The API key is read from the environment at call time, used only as the
client credential, and never logged or included in error messages.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
import os

from openai import APIConnectionError, InternalServerError, OpenAI, OpenAIError, RateLimitError

from src.config import (
    NOT_CONFIGURED_MESSAGE,
    generation_configured,
    generation_model,
    venice_base_url,
)

DEFAULT_MAX_TOKENS = 1500
# Venice answers 429 "model is currently overloaded" when a model is at capacity, which is
# unrelated to account credits. Retry a few times with backoff before giving up.
RETRY_DELAYS = (2.0, 5.0, 10.0)
_TRANSIENT = (RateLimitError, InternalServerError, APIConnectionError)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


class GenerationNotConfigured(RuntimeError):
    """Raised when VENICE_API_KEY is missing. Retrieval still works."""


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
    return OpenAI(api_key=os.environ["VENICE_API_KEY"], base_url=venice_base_url())


def generate(
    messages: list[dict[str, str]],
    model_id: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    client = _client()
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
        except _TRANSIENT as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise RuntimeError(
                f"Venice is busy or overloaded ({type(exc).__name__}) after {attempt + 1} attempts. "
                "This is a provider capacity limit, not your credits; try again or switch VENICE_MODEL."
            ) from None
        except OpenAIError as exc:
            # Deliberately drop the SDK message: it can echo request details.
            raise RuntimeError(f"Venice generation request failed ({type(exc).__name__}).") from None

    choice = response.choices[0]
    content = choice.message.content
    text = _THINK_BLOCK.sub("", content).strip() if isinstance(content, str) else ""
    if not text:
        reason = getattr(choice, "finish_reason", None)
        used = getattr(getattr(response, "usage", None), "completion_tokens", None)
        raise RuntimeError(f"Venice returned an empty text response (finish_reason={reason}, "
                           f"completion_tokens={used}, max_tokens={max_tokens}).")

    usage = getattr(response, "usage", None)
    return GenerationResult(
        text=text,
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
