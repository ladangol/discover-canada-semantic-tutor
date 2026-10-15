"""Daily budget, per-visitor caps and graceful provider-error handling. No network, no real key."""
from __future__ import annotations

import httpx
import openai
import pytest

from src import budget, generation
from src.generation import GenerationUnavailable, classify_error, generate


def status_error(code: int, body: dict | None = None, cls=openai.APIStatusError):
    resp = httpx.Response(code, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))
    return cls("provider said no", response=resp, body=body)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def create(self, **_):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeClient:
    def __init__(self, outcomes):
        self.chat = type("C", (), {"completions": FakeCompletions(outcomes)})()


def ok_response(pt=1000, ct=500, text="hello"):
    msg = type("M", (), {"content": text})()
    choice = type("Ch", (), {"message": msg, "finish_reason": "stop"})()
    usage = type("U", (), {"prompt_tokens": pt, "completion_tokens": ct})()
    return type("R", (), {"choices": [choice], "usage": usage})()


@pytest.fixture
def with_client(monkeypatch):
    def install(outcomes):
        client = FakeClient(outcomes)
        monkeypatch.setattr(generation, "_client", lambda: client)
        monkeypatch.setattr(generation, "RETRY_DELAYS", (0.0, 0.0, 0.0))
        return client.chat.completions
    return install


# --- error classification ---------------------------------------------------------------------------------

@pytest.mark.parametrize("code,body,expected", [
    (402, {"code": "INSUFFICIENT_BALANCE"}, ("budget", False)),
    (402, {"code": "API_KEY_USD_SPEND_LIMIT_EXCEEDED"}, ("budget", False)),
    (429, {"code": "MODEL_OVERLOADED"}, ("busy", True)),
    (429, {"code": "RATE_LIMIT_EXCEEDED"}, ("busy", True)),
    (503, None, ("busy", True)),
    (401, None, ("config", False)),
    (422, {"code": "CONTENT_POLICY_VIOLATION"}, ("policy", False)),
])
def test_provider_errors_map_to_reason_and_retryability(code, body, expected):
    assert classify_error(status_error(code, body)) == expected


# --- graceful behaviour in generate() ---------------------------------------------------------------------

def test_quota_error_is_not_retried_trips_the_ledger_and_blocks_later_calls(with_client, monkeypatch):
    comp = with_client([status_error(402, {"code": "API_KEY_USD_SPEND_LIMIT_EXCEEDED"}), ok_response()])
    with pytest.raises(GenerationUnavailable) as err:
        generate([{"role": "user", "content": "x"}])
    assert err.value.reason == "budget" and comp.calls == 1                      # no wasted retries
    assert "today" in str(err.value) and "limit_exceeded" not in str(err.value).lower()
    assert budget.status().exhausted and budget.status().reason == "provider"
    with pytest.raises(GenerationUnavailable):                                   # fails fast, provider not called
        generate([{"role": "user", "content": "x"}])
    assert comp.calls == 1


def test_overload_is_retried_then_succeeds(with_client):
    comp = with_client([status_error(429, {"code": "MODEL_OVERLOADED"}, openai.RateLimitError), ok_response()])
    assert generate([{"role": "user", "content": "x"}]).text == "hello" and comp.calls == 2


def test_persistent_overload_gives_a_friendly_busy_message(with_client):
    comp = with_client([status_error(429, {"code": "MODEL_OVERLOADED"}, openai.RateLimitError)] * 4)
    with pytest.raises(GenerationUnavailable) as err:
        generate([{"role": "user", "content": "x"}])
    assert err.value.reason == "busy" and comp.calls == 4 and "try again" in str(err.value).lower()


def test_error_messages_never_contain_provider_text_or_keys(with_client, monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "sk-secret-should-never-appear")
    with_client([status_error(500, {"error": "internal detail sk-secret-should-never-appear"})] * 4)
    with pytest.raises(GenerationUnavailable) as err:
        generate([{"role": "user", "content": "x"}])
    assert "secret" not in str(err.value) and "internal detail" not in str(err.value)


# --- daily budget ------------------------------------------------------------------------------------------

def test_token_budget_stops_calls_after_the_limit(with_client, monkeypatch):
    monkeypatch.setenv("STUDY_DAILY_TOKEN_BUDGET", "2000")
    comp = with_client([ok_response(1000, 500), ok_response(1000, 500), ok_response()])
    generate([{"role": "user", "content": "a"}])                                 # 1,500 tokens
    generate([{"role": "user", "content": "b"}])                                 # 3,000 total >= 2,000
    with pytest.raises(GenerationUnavailable) as err:
        generate([{"role": "user", "content": "c"}])
    assert err.value.reason == "budget" and comp.calls == 2


def test_usd_budget_needs_prices_and_enforces_one_dollar_a_day(with_client, monkeypatch):
    monkeypatch.setenv("STUDY_DAILY_USD_BUDGET", "1.00")
    monkeypatch.setenv("STUDY_DAILY_TOKEN_BUDGET", "0")
    assert not budget.status().usd_enforced                                      # no prices -> not enforced
    monkeypatch.setenv("VENICE_PRICE_IN_PER_M", "1.00")
    monkeypatch.setenv("VENICE_PRICE_OUT_PER_M", "4.00")
    assert budget.status().usd_enforced
    comp = with_client([ok_response(600_000, 100_000), ok_response()])           # $0.60 + $0.40 = $1.00
    generate([{"role": "user", "content": "a"}])
    assert budget.status().usd == pytest.approx(1.00) and budget.status().exhausted
    with pytest.raises(GenerationUnavailable):
        generate([{"role": "user", "content": "b"}])
    assert comp.calls == 1


def test_counters_reset_on_a_new_day(monkeypatch):
    monkeypatch.setenv("STUDY_DAILY_TOKEN_BUDGET", "10")
    budget.record(100, 100)
    assert budget.status().exhausted
    monkeypatch.setattr(budget, "today", lambda: "2099-01-02")
    assert not budget.status().exhausted


def test_visitor_counters_add_and_refund():
    assert budget.add_visitor_calls("visitor:abc", 3) == 3
    assert budget.add_visitor_calls("visitor:abc", -1) == 2 and budget.visitor_calls("visitor:other") == 0


def test_cached_results_never_touch_the_budget(monkeypatch):
    """A cache hit returns before any generate() call, so it works even when the allowance is gone."""
    from src import learning_extractors as ex
    from src.corpus import load_corpus
    from src import config
    import src.learning_cache as lc, tempfile, pathlib

    monkeypatch.setattr(config, "GENERATED_DIR", pathlib.Path(tempfile.mkdtemp()))
    corpus = load_corpus()
    calls = []

    def fake(messages, **_):
        calls.append(1)
        from src.generation import GenerationResult
        return GenerationResult('{"central_concept":"x","summary":"","summary_sources":[],"nodes":[]}', "stop", 1, 1)

    ex.build_concept_hub("Prime Minister", corpus, gen=fake)                     # builds and caches
    budget.trip()                                                                # allowance gone
    hub, cached = ex.build_concept_hub("Prime Minister", corpus, gen=fake)       # served from cache
    assert cached and len(calls) == 1


def test_library_only_mode_never_builds_a_new_timeline():
    from src import learning_extractors as ex
    from src.corpus import load_corpus
    with pytest.raises(GenerationUnavailable) as err:
        ex.build_timeline(load_corpus(), topic="a topic nobody pre-built", cache_only=True)
    assert err.value.reason == "library" and "ready-made" in str(err.value)
