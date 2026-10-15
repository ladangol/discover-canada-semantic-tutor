"""Tell the page's CSS whether Streamlit's active theme is light or dark (set <html data-mode=...>).

The server cannot know this reliably on the first run, so an invisible component reads the active
theme's text colour in the browser and keeps `data-mode` up to date when the user switches theme.
"""
from __future__ import annotations

import streamlit as st

_JS = """
export default function () {
  const apply = () => {
    const app = document.querySelector('.stApp') || document.body;
    const m = (getComputedStyle(app).color.match(/\\d+(\\.\\d+)?/g) || [0, 0, 0]).map(Number);
    const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;   // light text => dark theme
    document.documentElement.dataset.mode = lum > 0.55 ? 'dark' : 'light';
  };
  apply();
  const mo = new MutationObserver(apply);
  mo.observe(document.documentElement, { attributes: true, subtree: true, attributeFilter: ['class', 'style', 'data-theme'] });
  const t = setInterval(apply, 1500);                 // cheap safety net for theme switches
  return () => { mo.disconnect(); clearInterval(t); };
}
"""

_component = st.components.v2.component("discover_canada_theme_sync", html="<span></span>", css="span{display:none}", js=_JS)


def sync() -> None:
    _component(key="theme_sync")
