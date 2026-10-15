"""Server-wide daily usage ledger: the app-side spending cap for a public deployment.

Why this exists: the provider key can carry its own spend limit (belt), but the app must also stop itself
(braces) and fail gracefully. One small SQLite file (WAL mode) is shared by every Streamlit session and
process on the box. Counters reset at local midnight (STUDY_TZ, default America/Toronto).

Settings (environment):
  STUDY_DAILY_USD_BUDGET     dollars per day, default 1.00 (enforced only if both prices below are set)
  VENICE_PRICE_IN_PER_M      $ per 1M prompt tokens for the chosen model (look it up on Venice; unset = unknown)
  VENICE_PRICE_OUT_PER_M     $ per 1M completion tokens
  STUDY_DAILY_TOKEN_BUDGET   safety net that works without prices, default 500000 tokens/day (0 = off)
  STUDY_DATA_DIR             where usage.sqlite3 lives, default ./data
  STUDY_TZ                   day boundary timezone
No prompts, answers, keys or IP addresses are stored (visitors are a salted per-day hash).
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import ROOT


class DailyBudgetExceeded(RuntimeError):
    """Today's AI allowance is used up (or the provider said its limit was hit)."""


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def today() -> str:
    return datetime.now(ZoneInfo(os.environ.get("STUDY_TZ", "America/Toronto"))).date().isoformat()


def _db() -> sqlite3.Connection:
    path = Path(os.environ.get("STUDY_DATA_DIR") or ROOT / "data") / "usage.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS usage (
        day TEXT NOT NULL, scope TEXT NOT NULL, calls INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
        usd REAL NOT NULL DEFAULT 0, tripped INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (day, scope))""")
    return con


def _bump(scope: str, *, calls: int = 0, tokens: int = 0, usd: float = 0.0, tripped: int = 0) -> None:
    with _db() as con:
        con.execute("INSERT INTO usage (day, scope, calls, tokens, usd, tripped) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(day, scope) DO UPDATE SET calls=MAX(0, calls+excluded.calls), "
                    "tokens=tokens+excluded.tokens, usd=usd+excluded.usd, tripped=MAX(tripped, excluded.tripped)",
                    (today(), scope, calls, tokens, usd, tripped))


def _row(scope: str) -> tuple[int, int, float, int]:
    with _db() as con:
        r = con.execute("SELECT calls, tokens, usd, tripped FROM usage WHERE day=? AND scope=?",
                        (today(), scope)).fetchone()
    return r or (0, 0, 0.0, 0)


def prices() -> tuple[float, float] | None:
    pin, pout = _float("VENICE_PRICE_IN_PER_M", 0), _float("VENICE_PRICE_OUT_PER_M", 0)
    return (pin, pout) if pin > 0 and pout > 0 else None


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    p = prices()
    return 0.0 if p is None else prompt_tokens * p[0] / 1e6 + completion_tokens * p[1] / 1e6


@dataclass(frozen=True)
class Status:
    exhausted: bool
    reason: str            # "" | "usd" | "tokens" | "provider"
    calls: int
    tokens: int
    usd: float
    usd_cap: float
    usd_enforced: bool


def status() -> Status:
    calls, tokens, usd, tripped = _row("global")
    usd_cap, tok_cap = _float("STUDY_DAILY_USD_BUDGET", 1.0), _float("STUDY_DAILY_TOKEN_BUDGET", 500_000)
    enforced = prices() is not None and usd_cap > 0
    reason = ("provider" if tripped else "usd" if enforced and usd >= usd_cap
              else "tokens" if tok_cap > 0 and tokens >= tok_cap else "")
    return Status(bool(reason), reason, calls, tokens, usd, usd_cap, enforced)


def check() -> None:
    """Raise before spending anything if today's allowance is gone."""
    st = status()
    if st.exhausted:
        raise DailyBudgetExceeded(st.reason)


def record(prompt_tokens: int, completion_tokens: int) -> None:
    _bump("global", calls=1, tokens=prompt_tokens + completion_tokens, usd=cost_usd(prompt_tokens, completion_tokens))


def trip() -> None:
    """The provider refused (balance / key spend limit): stop calling it for the rest of the day."""
    _bump("global", tripped=1)


# --- per-visitor daily action counters (abuse control) --------------------------------------------

def visitor_calls(scope: str) -> int:
    return _row(scope)[0]


def add_visitor_calls(scope: str, n: int) -> int:
    _bump(scope, calls=n)
    return _row(scope)[0]
