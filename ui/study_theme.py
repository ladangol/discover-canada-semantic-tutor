"""Look and feel of Discover Canada Study: palette, CSS, and small HTML building blocks.

Colours come from two palettes chosen by the active Streamlit theme (light/dark), so the custom
parts follow the same switch as the native widgets. Only the learner app injects this CSS.
"""
from __future__ import annotations

import html

import streamlit as st

# page -> (label, material icon, accent)
PAGES = {
    "Learn": ("Learn", ":material/school:", "#5B5BD6"),
    "Practice": ("Practice", ":material/bolt:", "#F2542D"),
    "Flashcards": ("Flashcards", ":material/style:", "#10A89A"),
    "Concept map": ("Concept map", ":material/hub:", "#8B5CF6"),
    "Timeline": ("Timeline", ":material/timeline:", "#E89A0C"),
}

# chapter -> (emoji, gradient start, gradient end)
CHAPTER_STYLE = {
    "Understanding the Oath": ("handshake", "#5B4BE6", "#9B87F5"),
    "Rights and Responsibilities of Citizenship": ("verified_user", "#1D4ED8", "#38BDF8"),
    "Who We Are": ("groups", "#BE185D", "#FB7185"),
    "Canada’s History": ("history_edu", "#B45309", "#F5A524"),
    "Modern Canada": ("location_city", "#0E7490", "#22D3EE"),
    "How Canadians Govern Themselves": ("account_balance", "#4338CA", "#818CF8"),
    "Federal Elections": ("how_to_vote", "#C2410C", "#FB923C"),
    "The Justice System": ("gavel", "#0F766E", "#2DD4BF"),
    "Canadian Symbols": ("flag", "#D0021B", "#FF6B6B"),
    "Canada’s Economy": ("trending_up", "#15803D", "#4ADE80"),
    "Canada’s Regions": ("map", "#6D28D9", "#C084FC"),
}
DEFAULT_STYLE = ("menu_book", "#475569", "#94A3B8")

LIGHT = dict(bg="#FFF8F0", surface="#FFFFFF", ink="#1B1B3A", muted="#6A6A8E", line="#EBDCCB",
             shadow="rgba(27,27,58,.10)", g1="#E4002B", g2="#FF5A36", g3="#FFA62B", chip="rgba(255,255,255,.22)",
             soft="#FBEFE2")
DARK = dict(bg="#0E1020", surface="#181B33", ink="#F4F4FF", muted="#A5A9CC", line="#2C3057",
            shadow="rgba(0,0,0,.45)", g1="#C2003A", g2="#7B2FF7", g3="#2563EB", chip="rgba(255,255,255,.14)",
            soft="#20244A")


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:          # old Streamlit or no session context
        return False


def _nav_css() -> str:
    rules = []
    for name, (_, _, accent) in PAGES.items():
        k = name.lower().replace(" ", "_")
        rules.append(f".st-key-nav_{k} button, .st-key-nav_{k}_on button {{ --accent: {accent}; }}")
    return "\n".join(rules)


def _chapter_css() -> str:
    rules = []
    for i, (_, (_, a, b)) in enumerate(CHAPTER_STYLE.items()):
        rules.append(
            f'.st-key-tile_{i} button {{ background: radial-gradient(120% 90% at 100% 0%, rgba(255,255,255,.28), transparent 55%), '
            f'linear-gradient(140deg, {a}, {b}) !important; }}\n'
            f'.st-key-tile_{i} button::after {{ content: "{i + 1:02d}"; }}')
    return "\n".join(rules)


def _vars(p: dict) -> str:
    return (f"--bg:{p['bg']}; --surface:{p['surface']}; --ink:{p['ink']}; --muted:{p['muted']}; --line:{p['line']}; "
            f"--shadow:{p['shadow']}; --g1:{p['g1']}; --g2:{p['g2']}; --g3:{p['g3']}; --chip:{p['chip']}; --soft:{p['soft']};")


def css() -> str:
    """One stylesheet for both modes: <html data-mode="dark|light"> (set by ui/theme_sync.py) picks the palette;
    prefers-color-scheme is only the first-paint fallback."""
    return f"""
<style>
:root {{ {_vars(LIGHT)}
  --display:'Bricolage Grotesque','Plus Jakarta Sans',system-ui,sans-serif; --ease:cubic-bezier(.2,.8,.2,1); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-mode]) {{ {_vars(DARK)} }} }}
:root[data-mode="dark"] {{ {_vars(DARK)} }}
:root[data-mode="light"] {{ {_vars(LIGHT)} }}

/* ---------- page frame ---------- */
.stApp {{ background:
   radial-gradient(900px 420px at 8% -5%, color-mix(in srgb, var(--g2) 14%, transparent), transparent 60%),
   radial-gradient(800px 380px at 100% 0%, color-mix(in srgb, var(--g3) 12%, transparent), transparent 60%),
   var(--bg) !important; }}
[data-testid="stHeader"], [data-testid="stHeader"] .stAppToolbar {{ background: transparent; pointer-events: none !important; }}
[data-testid="stHeader"] button, [data-testid="stHeader"] a, [data-testid="stMainMenu"], [data-testid="stToolbarActions"] {{ pointer-events: auto !important; }}

[data-testid="stMainBlockContainer"] {{ max-width: 1100px; padding: 1.2rem clamp(.8rem, 3vw, 2rem) 4rem; }}
h1, h2, h3, h4 {{ font-family: var(--display) !important; letter-spacing: -.01em; }}
.stApp {{ font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}

/* ---------- hero: brand line + mascot with a "did you know" bubble ---------- */
.hero {{ position: relative; overflow: hidden; border-radius: 30px; color: #fff; margin: .2rem 0 .6rem;
  padding: clamp(16px, 2.6vw, 26px) clamp(18px, 3.6vw, 40px) clamp(18px, 3vw, 30px);
  background: linear-gradient(120deg, var(--g1), var(--g2) 55%, var(--g3)); background-size: 220% 220%;
  animation: flow 16s ease infinite, rise .7s var(--ease) both; box-shadow: 0 22px 50px -18px var(--shadow); }}
.brand {{ display: flex; align-items: center; gap: .7rem; flex-wrap: wrap; position: relative; z-index: 2; }}
.mi {{ font-family: 'Material Symbols Rounded'; font-weight: 400; font-style: normal; font-size: 1.15em; line-height: 1; display: inline-block;
  vertical-align: -.2em; font-feature-settings: 'liga'; font-variation-settings: 'FILL' 1, 'wght' 500; -webkit-font-smoothing: antialiased; }}
.brand .leaf {{ width: 38px; height: 38px; flex: none; filter: drop-shadow(0 2px 6px rgba(0,0,0,.25)); }}
.brand .leaf {{ animation: sway 6s ease-in-out infinite; transform-origin: 50% 90%; }}
.brand b {{ font-family: var(--display); font-size: clamp(1.25rem, 3vw, 1.7rem); font-weight: 800; letter-spacing: -.01em; }}
.brand .tag {{ margin-left: auto; font-size: .78rem; font-weight: 700; padding: .3rem .75rem; border-radius: 999px;
  background: var(--chip); border: 1px solid rgba(255,255,255,.28); backdrop-filter: blur(8px); white-space: nowrap; }}
.stage {{ display: flex; align-items: center; gap: clamp(12px, 3vw, 34px); margin-top: .7rem; position: relative; z-index: 2; }}
.mascot {{ width: clamp(110px, 17vw, 190px); height: auto; flex: none; transform-origin: 50% 90%;
  filter: drop-shadow(0 14px 14px rgba(0,0,0,.28)); animation: wave 5s ease-in-out infinite; }}
.mascot:hover {{ animation: hop .6s var(--ease); }}
.bubble {{ position: relative; background: var(--surface); color: var(--ink); border-radius: 26px; flex: 1; min-width: 0;
  padding: clamp(.9rem, 2vw, 1.3rem) clamp(1rem, 2.4vw, 1.6rem); box-shadow: 0 18px 34px -16px rgba(0,0,0,.4);
  animation: pop .6s var(--ease) both; }}
.bubble::before {{ content: ""; position: absolute; left: -12px; top: 42%; width: 26px; height: 26px; background: var(--surface);
  transform: rotate(45deg); border-radius: 6px; }}
.bubble small {{ display: block; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; font-size: .74rem; color: var(--g1); margin-bottom: .25rem; }}
.bubble p {{ margin: 0; font-family: var(--display); font-weight: 700; font-size: clamp(1.05rem, 2.5vw, 1.45rem); line-height: 1.25; }}
.bubble cite {{ display: block; margin-top: .45rem; font-style: normal; font-size: .8rem; color: var(--muted); font-weight: 600; }}
.blob {{ position: absolute; border-radius: 50%; pointer-events: none; animation: float 10s ease-in-out infinite; }}
.blob.a {{ width: 260px; height: 260px; right: -70px; top: -90px; background: rgba(255,255,255,.16); }}
.blob.b {{ width: 140px; height: 140px; left: 34%; bottom: -70px; background: rgba(255,255,255,.12); animation-delay: -4s; }}
@media (max-width: 640px) {{
  .stage {{ flex-direction: column; align-items: stretch; gap: .4rem; }}
  .mascot {{ width: 104px; align-self: center; margin-bottom: -.2rem; }}
  .bubble::before {{ left: 46%; top: -12px; }}
  .brand .tag {{ margin-left: 0; }}
}}

/* ---------- navigation ---------- */
.st-key-nav {{ position: sticky; top: .4rem; z-index: 60; padding: .55rem; margin: 0 0 1.2rem; border-radius: 26px;
  background: color-mix(in srgb, var(--surface) 82%, transparent); backdrop-filter: blur(14px);
  border: 1px solid var(--line); box-shadow: 0 10px 30px -14px var(--shadow); }}
.st-key-nav [data-testid="stHorizontalBlock"] {{ gap: .5rem; flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; }}
.st-key-nav [data-testid="stColumn"] {{ min-width: 92px; }}
[class*="st-key-nav_"] button {{ width: 100%; border-radius: 18px !important; padding: .7rem .6rem; font-weight: 700;
  background: transparent; border: 1.5px solid transparent; color: var(--ink);
  transition: transform .22s var(--ease), background .22s, box-shadow .22s, color .22s; }}
[class*="st-key-nav_"] button:hover {{ background: color-mix(in srgb, var(--accent) 14%, transparent); transform: translateY(-2px);
  color: var(--accent); }}
[class*="st-key-nav_"] button:active {{ transform: scale(.97); }}
[class*="st-key-nav_"][class*="_on"] button {{ background: var(--accent); color: #fff !important; border-color: var(--accent);
  box-shadow: 0 10px 22px -10px var(--accent); }}
@media (min-width: 641px) and (max-width: 900px) {{
  [class*="st-key-nav_"] button {{ font-size: .82rem; padding: .65rem .3rem; gap: .25rem; }}
  .st-key-nav [data-testid="stColumn"] {{ min-width: 0 !important; }}
  .st-key-tiles [data-testid="stColumn"] {{ min-width: calc(33.33% - .75rem); }}
}}
@media (max-width: 640px) {{
  .st-key-nav {{ padding: .4rem; border-radius: 22px; }}
  .st-key-nav [data-testid="stHorizontalBlock"] {{ gap: .3rem; }}
  .st-key-nav [data-testid="stColumn"] {{ min-width: 0 !important; flex: 1 1 0 !important; }}
  .st-key-nav [data-testid="stColumn"]:has([class*="_on"]) {{ flex: 2.6 1 0 !important; }}
  [class*="st-key-nav_"] button {{ padding: .6rem .25rem; gap: .3rem; }}
  [class*="st-key-nav_"] button p {{ display: none; }}
  [class*="st-key-nav_"][class*="_on"] button p {{ display: block; font-size: .85rem; }}
  [class*="st-key-tile_"] button {{ min-height: 128px; padding: .75rem .7rem; border-radius: 22px !important; }}
  [class*="st-key-tile_"] button span[data-has-shortcut] {{ min-height: 100px; gap: 1rem; }}
  [class*="st-key-tile_"] button p {{ font-size: .8rem; max-width: 100%; hyphens: none; overflow-wrap: normal; letter-spacing: -.015em; }}
  [class*="st-key-tile_"] button::after {{ font-size: 2.6rem; }}
}}

/* ---------- section headers & cards ---------- */
.sec {{ display: flex; align-items: center; gap: .9rem; margin: .4rem 0 1rem; animation: rise .5s var(--ease) both; }}
.sec .badge {{ width: 52px; height: 52px; border-radius: 18px; display: grid; place-items: center; font-size: 1.7rem;
  color: #fff; background: var(--accent); box-shadow: 0 12px 22px -10px var(--accent); flex: none; }}
.sec h2 {{ margin: 0; padding: 0; font-size: clamp(1.5rem, 3.6vw, 2.2rem); }}
.sec p {{ margin: .1rem 0 0; color: var(--muted); }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .answer-card) {{ border: 0; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 22px; padding: 1.1rem 1.3rem;
  box-shadow: 0 12px 30px -18px var(--shadow); animation: rise .5s var(--ease) both;
  border-left: 6px solid var(--accent, var(--g1)); }}
.pills span {{ display: inline-block; padding: .25rem .7rem; margin: .15rem .3rem .15rem 0; border-radius: 999px;
  background: color-mix(in srgb, var(--accent, var(--g1)) 14%, transparent); color: var(--accent, var(--g1));
  font-weight: 700; font-size: .82rem; }}

.ans-head {{ display: flex; align-items: center; gap: .6rem; margin-bottom: .4rem; font-family: var(--display); font-weight: 800; color: var(--muted); }}
.ans-head img {{ width: 40px; height: 40px; object-fit: contain; }}
.empty img {{ width: 92px; height: auto; display: block; margin: 0 auto .6rem; animation: wave 5s ease-in-out infinite; filter: drop-shadow(0 8px 8px rgba(0,0,0,.2)); }}
.st-key-answer {{ background: var(--surface); border: 1px solid var(--line); border-left: 6px solid #5B5BD6; border-radius: 22px;
  padding: 1rem 1.3rem; margin: .8rem 0; box-shadow: 0 14px 30px -20px var(--shadow); animation: rise .5s var(--ease) both; }}
[class*="st-key-chips_"] button {{ border-radius: 999px !important; background: var(--surface); border: 1.5px solid var(--line);
  padding: .35rem .9rem; transition: transform .2s var(--ease), border-color .2s, background .2s; }}
[class*="st-key-chips_"] button:hover {{ border-color: var(--g2); background: color-mix(in srgb, var(--g2) 12%, var(--surface)); transform: translateY(-2px); }}

/* ---------- buttons & inputs ---------- */
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{
  transition: transform .2s var(--ease), box-shadow .2s, filter .2s; font-weight: 700; }}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ transform: translateY(-2px);
  box-shadow: 0 10px 22px -10px var(--shadow); }}
.stButton button:active {{ transform: scale(.97); }}
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: linear-gradient(120deg, var(--g1), var(--g2)) !important; border: 0 !important; color: #fff !important;
  box-shadow: 0 12px 24px -12px var(--g1); }}
.stTextInput input, .stTextArea textarea {{ border-radius: 16px !important; }}
.stTextInput input:focus, .stTextArea textarea:focus {{ box-shadow: 0 0 0 3px color-mix(in srgb, var(--g2) 35%, transparent) !important; }}

/* ---------- chapter tiles ---------- */
[class*="st-key-tile_"] button {{ width: 100%; height: auto; min-height: 150px; border: 0 !important; border-radius: 26px !important; color: #fff !important;
  padding: 1.05rem 1.15rem; white-space: normal; position: relative; overflow: hidden; text-align: left;
  box-shadow: 0 18px 30px -18px var(--shadow), inset 0 1px 0 rgba(255,255,255,.25);
  transition: transform .3s var(--ease), box-shadow .3s, filter .3s; }}
/* big numeral watermark, top-right */
[class*="st-key-tile_"] button::after {{ position: absolute; right: 1rem; top: .35rem; font-family: var(--display); font-weight: 800;
  font-size: 3.4rem; line-height: 1; letter-spacing: -.05em; color: rgba(255,255,255,.26); transition: transform .35s var(--ease), color .3s; }}
/* arrow that slides in on hover, bottom-right */
[class*="st-key-tile_"] button::before {{ content: "\\2192"; position: absolute; right: 1.05rem; bottom: .85rem; font-size: 1.5rem; font-weight: 700;
  opacity: 0; transform: translateX(-10px); transition: opacity .25s, transform .3s var(--ease); }}
[class*="st-key-tile_"] button > div {{ width: 100%; height: 100%; }}
[class*="st-key-tile_"] button span[data-has-shortcut] {{ display: flex; flex-direction: column; justify-content: space-between;
  align-items: flex-start; gap: 1.4rem; width: 100%; min-height: 118px; }}
/* glass icon chip (its parent wrapper is 16px wide by default, so release it) */
[class*="st-key-tile_"] button span[data-has-shortcut] > span:first-child {{ width: auto !important; height: auto !important; flex: none !important; }}
[class*="st-key-tile_"] button [data-testid="stIconMaterial"] {{ flex: none !important; min-width: 2.75rem; min-height: 2.75rem; }}
/* glass icon chip */
[class*="st-key-tile_"] button [data-testid="stIconMaterial"] {{ font-size: 1.55rem !important; width: 2.75rem; height: 2.75rem; display: grid; place-items: center;
  border-radius: 15px; background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.32); backdrop-filter: blur(6px); color: #fff; }}
[class*="st-key-tile_"] button p {{ font-family: var(--display); font-weight: 800; font-size: 1.12rem; line-height: 1.12; letter-spacing: -.01em; text-align: left;
  margin: 0; white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: break-word; hyphens: auto; max-width: 88%; }}
[class*="st-key-tile_"] button:hover {{ transform: translateY(-7px); filter: saturate(1.12) brightness(1.04);
  box-shadow: 0 28px 40px -18px var(--shadow), inset 0 1px 0 rgba(255,255,255,.3); }}
[class*="st-key-tile_"] button:hover::before {{ opacity: 1; transform: translateX(0); }}
[class*="st-key-tile_"] button:hover::after {{ color: rgba(255,255,255,.42); transform: translateY(2px) scale(1.06); }}
[data-testid="stAppDeployButton"], .stAppDeployButton {{ display: none; }}
.st-key-tiles [data-testid="stColumn"] {{ min-width: calc(50% - .75rem); }}
@media (min-width: 900px) {{ .st-key-tiles [data-testid="stColumn"] {{ min-width: calc(25% - .75rem); }} }}
@media (max-width: 640px) {{   /* must come AFTER the base tile rules above to win */
  [class*="st-key-tile_"] button {{ min-height: 128px; padding: .75rem .7rem; border-radius: 22px !important; }}
  [class*="st-key-tile_"] button span[data-has-shortcut] {{ min-height: 100px; gap: 1rem; }}
  [class*="st-key-tile_"] button p {{ font-size: .8rem; max-width: 100%; hyphens: none; overflow-wrap: normal; letter-spacing: -.015em; }}
  [class*="st-key-tile_"] button::after {{ font-size: 2.6rem; right: .7rem; }}
}}

/* ---------- quiz ---------- */
[class*="st-key-opt_"] button {{ width: 100%; justify-content: flex-start; text-align: left; min-height: 64px; padding: .8rem 1.1rem;
  border-radius: 20px !important; background: var(--surface); border: 2px solid var(--line); color: var(--ink);
  font-weight: 600; }}
[class*="st-key-opt_"] button p {{ text-align: left; font-size: 1.02rem; }}
[class*="st-key-opt_"] button:hover {{ border-color: var(--g2); transform: translateY(-3px); }}
[class*="st-key-opt_"][class*="_picked"] button {{ border-color: var(--g1); background: color-mix(in srgb, var(--g1) 10%, var(--surface));
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--g1) 25%, transparent); }}
[class*="st-key-opt_"][class*="_right"] button {{ border-color: #12A37F; background: color-mix(in srgb, #12A37F 16%, var(--surface));
  animation: pop .45s var(--ease); opacity: 1 !important; color: var(--ink) !important; }}
[class*="st-key-opt_"][class*="_wrong"] button {{ border-color: #E5484D; background: color-mix(in srgb, #E5484D 14%, var(--surface));
  animation: shake .4s; opacity: 1 !important; color: var(--ink) !important; }}
.banner {{ border-radius: 20px; padding: .9rem 1.2rem; font-weight: 700; margin: .6rem 0; animation: pop .5s var(--ease); }}
.banner.good {{ background: color-mix(in srgb, #12A37F 18%, var(--surface)); border: 2px solid #12A37F; }}
.banner.bad  {{ background: color-mix(in srgb, #E5484D 14%, var(--surface)); border: 2px solid #E5484D; }}
.question {{ font-family: var(--display); font-weight: 800; font-size: clamp(1.25rem, 3vw, 1.7rem); line-height: 1.25; margin: .2rem 0 1rem; }}

/* ---------- dashboard strip ---------- */
.stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1rem; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 22px; padding: .8rem 1rem;
  display: flex; align-items: center; gap: .8rem; box-shadow: 0 10px 24px -18px var(--shadow);
  transition: transform .25s var(--ease); animation: rise .5s var(--ease) both; }}
.stat:hover {{ transform: translateY(-3px); }}
.stat .ic {{ width: 44px; height: 44px; border-radius: 15px; display: grid; place-items: center; font-size: 1.45rem; flex: none;
  color: var(--c); background: color-mix(in srgb, var(--c) 16%, var(--surface)); }}
.stat b {{ font-family: var(--display); font-size: 1.6rem; line-height: 1; display: block; }}
.stat small {{ color: var(--muted); font-weight: 600; }}
.ring {{ width: 44px; height: 44px; border-radius: 50%; flex: none; display: grid; place-items: center; font-size: .78rem; font-weight: 800;
  background: conic-gradient(#12A37F calc(var(--p) * 1%), var(--soft) 0); position: relative; }}
.ring::after {{ content: attr(data-t); position: absolute; inset: 5px; border-radius: 50%; background: var(--surface); display: grid; place-items: center; }}
@media (max-width: 640px) {{ .stats {{ grid-template-columns: 1fr; }} .stat {{ padding: .6rem .8rem; }} }}

.empty {{ text-align: center; padding: 2.2rem 1rem; color: var(--muted); border: 2px dashed var(--line); border-radius: 24px; }}
.empty .big {{ font-size: 2.6rem; display: block; animation: bob 4s ease-in-out infinite; }}
.footer {{ text-align: center; color: var(--muted); font-size: .85rem; margin-top: 2.5rem; }}

/* ---------- keyframes ---------- */
@keyframes sway {{ 0%,100% {{ transform: rotate(-6deg); }} 50% {{ transform: rotate(6deg); }} }}
@keyframes wave {{ 0%,100% {{ transform: rotate(-2deg) translateY(0); }} 50% {{ transform: rotate(3deg) translateY(-8px); }} }}
@keyframes hop {{ 0% {{ transform: translateY(0) scale(1); }} 40% {{ transform: translateY(-22px) scale(1.06, .96); }} 100% {{ transform: translateY(0) scale(1); }} }}
@keyframes flow {{ 0%,100% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} }}
@keyframes float {{ 0%,100% {{ transform: translateY(0) scale(1); }} 50% {{ transform: translateY(18px) scale(1.06); }} }}
@keyframes bob {{ 0%,100% {{ margin-top: 0; }} 50% {{ margin-top: -9px; }} }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes pop {{ 0% {{ transform: scale(.94); opacity: 0; }} 60% {{ transform: scale(1.03); }} 100% {{ transform: scale(1); opacity: 1; }} }}
@keyframes shake {{ 0%,100% {{ transform: translateX(0); }} 25% {{ transform: translateX(-6px); }} 75% {{ transform: translateX(6px); }} }}

{_nav_css()}
{_chapter_css()}
</style>
"""


def inject() -> None:
    st.html(css())


# --- HTML building blocks (all text is escaped) -------------------------------------------------

MASCOTS = ("goose", "moose", "beaver")
STATIC = "app/static/mascots"      # served by Streamlit (enableStaticServing in .streamlit/config.toml)
LEAF = '<img class="leaf" src="app/static/leaf.svg" alt="">'   # original geometric maple-leaf mark (static/leaf.svg)


def mascot_url(name: str) -> str:
    return f"{STATIC}/{name}.png"


def hero(mascot: str, fact: str, page: str) -> str:
    """Brand line + mascot with a speech bubble. All text is escaped; `mascot` is one of MASCOTS."""
    name = mascot if mascot in MASCOTS else MASCOTS[0]
    return f"""
<div class="hero">
  <div class="blob a"></div><div class="blob b"></div>
  <div class="brand">{LEAF}<b>Discover Canada</b><span class="tag"><span class="mi">auto_awesome</span> powered by AI embeddings</span></div>
  <div class="stage">
    <img class="mascot" src="{mascot_url(name)}" alt="Friendly {name} mascot">
    <div class="bubble"><small><span class="mi">lightbulb</span> Did you know…</small><p>{html.escape(fact)}</p>
      <cite>Discover Canada, {html.escape(page)}</cite></div>
  </div>
</div>"""


def section_header(icon: str, title: str, text: str, accent: str) -> str:
    return (f'<div class="sec" style="--accent:{accent}"><div class="badge"><span class="mi">{icon}</span></div>'
            f"<div><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></div></div>")


def stats_strip(correct: int, total: int, streak: int) -> str:
    pct = round(100 * correct / total) if total else 0
    return f"""
<div class="stats">
  <div class="stat"><div class="ic" style="--c:#F2542D"><span class="mi">local_fire_department</span></div><div><b>{streak}</b><small>in a row</small></div></div>
  <div class="stat"><div class="ic" style="--c:#12A37F"><span class="mi">check_circle</span></div><div><b>{correct}/{total}</b><small>answered right</small></div></div>
  <div class="stat"><div class="ring" style="--p:{pct}" data-t="{pct}%"></div><div><b>{pct}%</b><small>accuracy</small></div></div>
</div>"""


def empty_state(mascot: str, text: str) -> str:
    return f'<div class="empty"><img src="{mascot_url(mascot)}" alt="">{html.escape(text)}</div>'


def pills(items: list[str], accent: str) -> str:
    return f'<div class="pills" style="--accent:{accent}">' + "".join(f"<span>{html.escape(i)}</span>" for i in items) + "</div>"


def card(inner_html: str, accent: str) -> str:
    """`inner_html` must already be safe (escaped / produced by trusted code)."""
    return f'<div class="card" style="--accent:{accent}">{inner_html}</div>'
