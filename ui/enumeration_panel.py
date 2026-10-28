"""'Scan the whole guide' panel (opt-in, STUDY_ENUMERATION=1). A separate action from asking a question: it reads passages, costs several
AI calls (shown first), and reports exactly how it was scoped. See src/enumeration.py."""
from __future__ import annotations

import streamlit as st

from src import enumeration as en
from src.generation import GenerationNotConfigured, GenerationUnavailable
from ui.session_limits import LIMIT_MESSAGE, charge, clean_topic, refund

QUICK, THOROUGH = "Quick", "Thorough"


def _pages(item: en.EnumItem) -> str:
    return ", ".join(sorted({e.page for e in item.evidence}, key=lambda p: (len(p), p)))


def _show(res: en.EnumerationResult) -> None:
    scope = f"Read {res.scanned} of {res.total_passages} passages in {res.calls} AI calls ({'every passage' if res.mode == 'thorough' else 'the passages that use your word'})."
    st.caption(scope + (f" Focus: {res.focus}." if res.focus else ""))
    st.markdown(f"**The guide words these as “{res.kind}”**")
    for i in res.exact:
        st.markdown(f"- **{i.name}** · p. {_pages(i)} — “{i.evidence[0].quote}”")
    if not res.exact:
        st.markdown("_Nothing in the guide is worded that way._")
    if res.worded_differently:
        st.markdown("**Also mentioned, worded differently** (listed separately, not counted as the same thing)")
        for i in res.worded_differently:
            word = f" the guide says “{i.stated_as}”," if i.stated_as else ""
            st.markdown(f"- **{i.name}** ·{word} p. {_pages(i)} — “{i.evidence[0].quote}”")
    st.caption("Each item has a quote copied from the guide. This finds what the guide's wording makes findable; something named without your word "
               "or a related word nearby can be missed.")
    if res.omitted:
        with st.expander(f"{len(res.omitted)} item(s) dropped by validation"):
            st.write("\n".join(f"- {o}" for o in res.omitted))


def render(corpus) -> None:
    with st.expander("Looking for a complete list? Scan the whole guide", icon=":material/manage_search:"):
        st.caption("Asking a question looks at the 5 best-matching passages, so it can miss items. This instead reads passages one by one. "
                   "It costs several AI calls, shown below, and the result is saved.")
        c1, c2 = st.columns([2, 2])
        kind = clean_topic(c1.text_input("What do you want listed?", placeholder="e.g. languages, provinces, prime ministers", key="enum_kind"))
        focus = c2.selectbox("Only one people? (optional)", ["", "First Nations", "Métis", "Inuit"], key="enum_focus",
                             format_func=lambda x: x or "No focus")
        mode_label = st.segmented_control("How thorough?", [QUICK, THOROUGH], default=QUICK, key="enum_mode") or QUICK
        mode = "thorough" if mode_label == THOROUGH else "quick"
        if kind and en.head_stem(kind):
            plan = en.plan_scan(corpus, kind, mode)
            st.info(f"Plan: read {plan.passages} passages ({plan.scope}) in {plan.calls} AI calls.")
        if st.button("Scan the guide", icon=":material/search:", key="enum_go", disabled=not (kind and en.head_stem(kind))):
            plan = en.plan_scan(corpus, kind, mode)
            if not charge(max(1, plan.calls)):
                st.warning(LIMIT_MESSAGE)
            else:
                bar = st.progress(0.0, text="Reading the guide…")
                try:
                    res, cached = en.scan_guide(kind, corpus, focus=focus, mode=mode,
                                                progress=lambda d, t: bar.progress(d / max(1, t), text=f"Reading the guide… {d}/{t}"))
                    if cached:
                        refund(max(1, plan.calls))              # a saved result costs nothing
                    st.session_state["enum_result"] = res
                except GenerationNotConfigured as exc:
                    refund(max(1, plan.calls)), st.warning(f"{exc} Scanning needs it.")
                except GenerationUnavailable as exc:
                    refund(max(1, plan.calls)), st.info(str(exc))
                except (RuntimeError, ValueError) as exc:
                    refund(max(1, plan.calls)), st.error(str(exc))
                bar.empty()
        if st.session_state.get("enum_result"):
            _show(st.session_state["enum_result"])
