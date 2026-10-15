"""Small helpers shared by the Streamlit apps."""
from __future__ import annotations

import streamlit as st

from src import tutor
from src.generation import GenerationNotConfigured, GenerationUnavailable


def generate_or_explain(fn, *args, **kwargs):
    """Call a generation function; show a friendly message for expected failures, never a traceback.

    Error text comes from our own code (no keys, no provider internals)."""
    try:
        with st.spinner("Working on it…"):
            return fn(*args, **kwargs)
    except GenerationNotConfigured as exc:
        st.warning(f"{exc} The study tools need it; browsing is unaffected.")
    except GenerationUnavailable as exc:
        # budget used up / provider busy / policy: a calm, specific message, never a traceback.
        (st.info if exc.reason == "budget" else st.warning)(str(exc), icon="🌙" if exc.reason == "budget" else None)
    except (tutor.QuizGenerationError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
    return None
