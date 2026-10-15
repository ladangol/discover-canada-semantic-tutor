"""Per-session safeguards for a future public deployment (LLM calls cost money; input is untrusted).

Streamlit session state is per browser session, which is enough to stop one visitor from running up
the bill. A real public deployment should add a global rate limit at the host/reverse proxy too.
"""
from __future__ import annotations

import hashlib
import os
import re

import streamlit as st

from src import budget

MAX_TOPIC_CHARS = 120
LIMIT_MESSAGE = ("You've reached the study limit for this session. Please come back later, "
                 "or open a new tab to continue.")


def max_actions() -> int:
    try:
        return max(1, int(os.environ.get("STUDY_MAX_ACTIONS", "40")))
    except ValueError:
        return 40


def clean_topic(text: str) -> str:
    """Trim, collapse whitespace, cap the length."""
    return re.sub(r"\s+", " ", text or "").strip()[:MAX_TOPIC_CHARS]


def max_actions_per_visitor_day() -> int:
    try:
        return max(1, int(os.environ.get("STUDY_MAX_ACTIONS_PER_VISITOR_DAY", "60")))
    except ValueError:
        return 60


def visitor_scope() -> str | None:
    """A salted per-day hash of the client address (never stored raw). Needs the reverse proxy to SET
    X-Forwarded-For to the real client address (see deploy/nginx-discover-study.conf)."""
    try:
        ip = (st.context.headers.get("X-Forwarded-For") or st.context.ip_address or "").split(",")[0].strip()
    except Exception:
        ip = ""
    return "visitor:" + hashlib.sha256(f"{budget.today()}:{ip}".encode()).hexdigest()[:16] if ip else None


def charge(cost: int = 1) -> bool:
    """Reserve `cost` generation actions against the session AND the visitor's daily cap."""
    used = st.session_state.get("_actions_used", 0)
    if used + cost > max_actions():
        return False
    scope = visitor_scope()
    if scope and budget.visitor_calls(scope) + cost > max_actions_per_visitor_day():
        return False
    st.session_state["_actions_used"] = used + cost
    if scope:
        budget.add_visitor_calls(scope, cost)
    return True


def refund(cost: int = 1) -> None:
    st.session_state["_actions_used"] = max(0, st.session_state.get("_actions_used", 0) - cost)
    scope = visitor_scope()
    if scope:
        budget.add_visitor_calls(scope, -cost)
