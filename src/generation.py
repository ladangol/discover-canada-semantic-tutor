"""The ONLY module that knows Venice exists.

    generate(messages, model_id=None) -> GenerationResult

The API key is read from the environment at call time, used only as the
client credential, and never logged or included in error messages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
import os

from openai import OpenAI, OpenAIError

from src.config import (
    NOT_CONFIGURED_MESSAGE,
    generation_configured,
    generation_model,
    venice_base_url,
)

DEFAULT_MAX_TOKENS = 1500
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
    try:
        response = client.chat.completions.create(
            model=model_id or generation_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"venice_parameters": {"include_venice_system_prompt": False}},
        )
    except OpenAIError as exc:
        # Deliberately drop the SDK message: it can echo request details.
        raise RuntimeError(f"Venice generation request failed ({type(exc).__name__}).") from None

    choice = response.choices[0]
    content = choice.message.content
    text = _THINK_BLOCK.sub("", content).strip() if isinstance(content, str) else ""
    if not text:
        raise RuntimeError("Venice returned an empty text response.")

    usage = getattr(response, "usage", None)
    return GenerationResult(
        text=text,
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
