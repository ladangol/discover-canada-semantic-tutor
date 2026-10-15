"""Flip-card deck component (3D flip, progress, keyboard) for the learner app."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS = """
.fc { max-width: 680px; margin: 0 auto; outline: none; font-family: inherit; color: var(--st-text-color); }
.top { display: flex; align-items: center; gap: .8rem; margin-bottom: .9rem; }
.bar { flex: 1; height: 10px; border-radius: 99px; background: color-mix(in srgb, var(--accent) 18%, transparent); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 55%, #fff));
         transition: width .5s cubic-bezier(.2,.8,.2,1); }
.count { font-weight: 800; font-size: .95rem; color: var(--accent); }
.scene { perspective: 1400px; cursor: pointer; }
.card3d { position: relative; height: clamp(250px, 44vw, 320px); transform-style: preserve-3d; transition: transform .7s cubic-bezier(.2,.8,.2,1); }
.card3d.flip { transform: rotateY(180deg); }
.scene:hover .card3d:not(.flip) { transform: rotateY(-5deg) rotateX(2deg) translateY(-4px); }
.face { position: absolute; inset: 0; backface-visibility: hidden; border-radius: 28px; padding: clamp(1.1rem, 4vw, 2rem);
        display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;
        box-shadow: 0 24px 44px -24px rgba(0,0,0,.45); }
.front { background: linear-gradient(140deg, var(--accent), color-mix(in srgb, var(--accent) 55%, #5B6CFF)); color: #fff; }
.back { transform: rotateY(180deg); background: var(--st-secondary-background-color); color: var(--st-text-color);
        border: 3px solid var(--accent); }
.tag { position: absolute; top: 1rem; left: 1.2rem; font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; opacity: .85; }
.txt { font-family: var(--st-heading-font, inherit); font-weight: 800; font-size: clamp(1.25rem, 3.6vw, 1.9rem); line-height: 1.25; }
.back .txt { font-weight: 700; font-size: clamp(1.1rem, 3vw, 1.5rem); }
.meta { position: absolute; bottom: 1rem; left: 0; right: 0; font-size: .8rem; opacity: .7; padding: 0 1rem; }
.ctrl { display: flex; gap: .55rem; justify-content: center; align-items: stretch; margin-top: 1.1rem; flex-wrap: wrap; }
.btn { border: 0; border-radius: 999px; padding: .7rem 1.15rem; font: inherit; font-weight: 800; cursor: pointer;
       color: var(--st-text-color); background: var(--st-secondary-background-color);
       border: 1.5px solid var(--st-border-color); transition: transform .2s cubic-bezier(.2,.8,.2,1), box-shadow .2s, filter .2s; }
.btn:hover { transform: translateY(-3px); box-shadow: 0 12px 22px -12px rgba(0,0,0,.4); }
.btn:active { transform: scale(.96); }
.btn.got { background: #12A37F; color: #fff; border-color: #12A37F; }
.btn.again { background: #F2542D; color: #fff; border-color: #F2542D; }
.btn.go { background: linear-gradient(120deg, var(--accent), color-mix(in srgb, var(--accent) 55%, #5B6CFF)); color: #fff; border: 0; margin-top: .6rem; }
.btn.nav { width: 3rem; padding: .7rem 0; }
.tally { display: flex; gap: .9rem; justify-content: center; align-items: center; margin-top: .9rem; font-weight: 700; flex-wrap: wrap; }
.tally .h { font-weight: 500; font-size: .82rem; opacity: .65; }
.done { text-align: center; padding: 2rem 1rem; border-radius: 28px; border: 2px dashed var(--st-border-color); }
.done .big { font-size: 3.2rem; animation: bob 2.4s ease-in-out infinite; }
.done h3 { margin: .2rem 0; font-family: var(--st-heading-font, inherit); }
@keyframes bob { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-8px) } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""

_component = st.components.v2.component(
    "discover_canada_flashcards",
    html='<div class="fc" tabindex="0"></div>',
    css=_CSS,
    js=(Path(__file__).parent / "flashcards.js").read_text(),
)


def flashcard_deck(cards: list[dict], *, key: str, accent: str = "#10A89A") -> None:
    """cards: [{"front": str, "back": str, "meta": str}]. Progress ("got it") is kept in the browser."""
    _component(data={"cards": cards, "accent": accent}, key=key)
