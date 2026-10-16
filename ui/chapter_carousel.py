"""3D chapter carousel for the learner app (active card in front, neighbours tilted back)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS = """
.cc { position: relative; max-width: 980px; margin: 0 auto; padding: .4rem 0 .2rem; outline: none; user-select: none;
      font-family: inherit; color: var(--st-text-color); --card: clamp(210px, 46vw, 290px); }
.stage { position: relative; height: calc(var(--card) * 1.08); perspective: 1300px; touch-action: pan-y; }
.card { box-sizing: border-box; position: absolute; left: 50%; top: 0; width: var(--card); height: calc(var(--card) * 1.04); margin-left: calc(var(--card) / -2);
        border-radius: 28px; padding: 1.15rem 1.2rem 1.1rem; display: flex; flex-direction: column; justify-content: space-between;
        color: #fff; cursor: pointer; overflow: hidden;
        background: radial-gradient(120% 90% at 100% 0%, rgba(255,255,255,.28), transparent 55%), linear-gradient(145deg, var(--a), var(--b));
        box-shadow: 0 26px 40px -22px rgba(0,0,0,.55);
        transform: translateX(calc(var(--d) * 58%)) translateZ(calc(var(--ad) * -90px)) rotateY(calc(var(--sign) * var(--ad) * -14deg))
                   scale(calc(1 - var(--ad) * .07));
        transition: transform .55s cubic-bezier(.2,.8,.2,1), opacity .4s, filter .4s; filter: saturate(.8) brightness(.92); }
.card.on { filter: none; cursor: pointer; box-shadow: 0 34px 50px -22px rgba(0,0,0,.6); }
.card.on:hover { transform: translateY(-6px) scale(1.02); }
.chip { width: 3.1rem; height: 3.1rem; border-radius: 16px; display: grid; place-items: center; background: rgba(255,255,255,.22);
        backdrop-filter: blur(6px); box-shadow: inset 0 0 0 1px rgba(255,255,255,.28); }
.chip .mi { font-size: 1.7rem; }
.num { position: absolute; right: 1rem; top: .3rem; font-weight: 800; font-size: 3.6rem; letter-spacing: -.05em; color: rgba(255,255,255,.26);
       font-family: var(--st-heading-font, inherit); line-height: 1; }
h3 { margin: 0; font-family: var(--st-heading-font, inherit); font-weight: 800; font-size: clamp(1.15rem, 3.4vw, 1.45rem);
     line-height: 1.1; letter-spacing: -.01em; text-wrap: balance; }
.go { font-weight: 800; font-size: .82rem; letter-spacing: .02em; opacity: 0; transform: translateY(6px); transition: opacity .3s, transform .3s; }
.card.on .go { opacity: .95; transform: none; }
.mi { font-family: 'Material Symbols Rounded'; font-weight: 400; font-style: normal; line-height: 1; display: inline-block;
      font-feature-settings: 'liga'; font-variation-settings: 'FILL' 1, 'wght' 500; vertical-align: -.2em; -webkit-font-smoothing: antialiased; }
.arrow { position: absolute; top: calc(var(--card) * .46); z-index: 200; width: 2.9rem; height: 2.9rem; border-radius: 50%; border: 0; cursor: pointer;
         display: grid; place-items: center; color: var(--st-text-color); background: var(--st-secondary-background-color);
         box-shadow: 0 10px 24px -10px rgba(0,0,0,.5), inset 0 0 0 1.5px var(--st-border-color); transition: transform .2s, opacity .2s; }
.arrow .mi { font-size: 1.8rem; }
.arrow:hover:not(:disabled) { transform: scale(1.1); } .arrow:disabled { opacity: .3; cursor: default; }
.prev { left: 0; } .next { right: 0; }
.dots { display: flex; gap: .4rem; justify-content: center; margin-top: .9rem; flex-wrap: wrap; }
.dot { width: .55rem; height: .55rem; padding: 0; border-radius: 99px; border: 0; cursor: pointer; background: var(--st-border-color);
       transition: width .3s cubic-bezier(.2,.8,.2,1), background .3s; }
.dot.on { width: 1.6rem; background: var(--st-primary-color); }
.cta { display: flex; align-items: center; justify-content: center; gap: .25rem; margin: .9rem auto 0; padding: .75rem 1.4rem; border: 0;
       border-radius: 999px; font: inherit; font-weight: 800; color: #fff; cursor: pointer; max-width: 100%;
       background: linear-gradient(120deg, #E4002B, #FF5A36 60%, #FFA62B); box-shadow: 0 14px 26px -14px rgba(228,0,43,.7);
       transition: transform .2s, box-shadow .2s; }
.cta:hover { transform: translateY(-3px); } .cta:active { transform: scale(.97); }
@media (max-width: 560px) { .arrow { width: 2.4rem; height: 2.4rem; top: calc(var(--card) * .5); } .card { transform: translateX(calc(var(--d) * 46%)) translateZ(calc(var(--ad) * -90px)) rotateY(calc(var(--sign) * var(--ad) * -14deg)) scale(calc(1 - var(--ad) * .07)); } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

_component = st.components.v2.component(
    "discover_canada_chapter_carousel",
    html='<div class="cc" role="group" aria-label="Chapters"></div>',
    css=_CSS,
    js=(Path(__file__).parent / "chapter_carousel.js").read_text(),
)


def chapter_carousel(items: list[dict], *, key: str) -> str | None:
    """items: [{"title", "icon" (Material Symbols name), "a", "b" (gradient colours)}]. Returns the chapter the learner opened."""
    res = _component(data={"items": items}, key=key, on_open_change=lambda: None)
    return res.open
