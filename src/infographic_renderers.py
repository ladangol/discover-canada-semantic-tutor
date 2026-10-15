"""Scene -> SVG, and SVG -> PNG. Also the one-call export helpers.

All text is XML-escaped, so labels (which originate from an LLM) can never inject markup.
The interactive web view uses `interactive=True` (adds data-id hooks and CSS classes); exports
are clean: no data attributes, no scripts, no Streamlit chrome.
"""
from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from src.infographic_layout import (
    Scene, Text, layout_hub, layout_timeline,
)
from src.infographic_models import HubSpec, TimelineSpec
from src.infographic_themes import CANVASES, FONT_STACK, THEMES, Theme


def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _text(t: Text) -> str:
    weight = f' font-weight="{t.weight}"' if t.weight != 400 else ""
    anchor = f' text-anchor="{t.anchor}"' if t.anchor != "start" else ""
    return (f'<text x="{_f(t.x)}" y="{_f(t.y)}" font-size="{_f(t.size)}"{weight}{anchor} '
            f'fill="{t.fill}">{escape(t.text)}</text>')


def scene_to_svg(scene: Scene, *, interactive: bool = False, responsive: bool = False) -> str:
    """Serialize a scene. `responsive` omits fixed width/height so CSS can scale it."""
    th: Theme = scene.theme
    W, H = scene.width, scene.height
    size = "" if responsive else f' width="{_f(W)}" height="{_f(H)}"'
    hook = (lambda i: f' data-id={quoteattr(i)}') if interactive else (lambda i: "")
    cls = (lambda c: f' class="{c}"') if interactive else (lambda c: "")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(W)} {_f(H)}"{size} '
        f'font-family={quoteattr(FONT_STACK)} role="img" aria-label={quoteattr(scene.title)}>',
        f"<title>{escape(scene.title)}</title>",
    ]
    if th.shadow:
        out.append('<defs><filter id="sh" x="-10%" y="-10%" width="120%" height="130%">'
                   '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.14"/>'
                   "</filter></defs>")
    out.append(f'<rect width="{_f(W)}" height="{_f(H)}" fill="{th.bg}"/>')
    for b in scene.boxes:
        r = b.rect
        out.append(f'<rect x="{_f(r.x)}" y="{_f(r.y)}" width="{_f(r.w)}" height="{_f(r.h)}" '
                   f'rx="{_f(b.radius)}" fill="{b.fill}"/>')
    for ln in scene.lines:
        out.append(f'<line x1="{_f(ln.x1)}" y1="{_f(ln.y1)}" x2="{_f(ln.x2)}" y2="{_f(ln.y2)}" '
                   f'stroke="{ln.stroke}" stroke-width="{_f(ln.width)}" stroke-opacity="{ln.opacity}" stroke-linecap="round"/>')
    for e in scene.edges:
        out.append(f'<g{cls("edge")}{hook(e.id)}><path d="{e.d}" fill="none" stroke="{e.stroke}" '
                   f'stroke-width="{_f(e.width)}" stroke-opacity="{e.opacity}" stroke-linecap="round"/></g>')
    for d in scene.dots:
        stroke = f' stroke="{d.stroke}" stroke-width="2"' if d.stroke else ""
        out.append(f'<circle cx="{_f(d.cx)}" cy="{_f(d.cy)}" r="{_f(d.r)}" fill="{d.fill}"{stroke}/>')
    for p in scene.pills:
        r = p.rect
        out.append(f'<g{cls("pill")}{hook(p.id)}><rect x="{_f(r.x)}" y="{_f(r.y)}" width="{_f(r.w)}" '
                   f'height="{_f(r.h)}" rx="{_f(r.h / 2 if r.h < 30 else 10)}" fill="{p.fill}" '
                   f'stroke="{p.stroke}" stroke-width="1.2"/>{"".join(_text(t) for t in p.texts)}</g>')
    for c in scene.cards:
        r = c.rect
        shadow = ' filter="url(#sh)"' if th.shadow and c.kind != "center" else ""
        bar = (f'<path d="M{_f(r.x + c.radius)},{_f(r.y + 0.6)} H{_f(r.x + 6)} V{_f(r.bottom - 0.6)} H{_f(r.x + c.radius)} '
               f'A{_f(c.radius)},{_f(c.radius)} 0 0 1 {_f(r.x)},{_f(r.bottom - c.radius)} V{_f(r.y + c.radius)} '
               f'A{_f(c.radius)},{_f(c.radius)} 0 0 1 {_f(r.x + c.radius)},{_f(r.y + 0.6)} Z" fill="{c.accent}"/>'
               if c.bar else "")
        out.append(f'<g{cls("card " + c.kind)}{hook(c.id)}><rect x="{_f(r.x)}" y="{_f(r.y)}" width="{_f(r.w)}" '
                   f'height="{_f(r.h)}" rx="{_f(c.radius)}" fill="{c.fill}" stroke="{c.stroke}" '
                   f'stroke-width="{1.2 if c.kind != "center" else 0}"{shadow}/>{bar}'
                   f'{"".join(_text(t) for t in c.texts)}</g>')
    out.append("".join(_text(t) for t in scene.texts))
    out.append("</svg>")
    return "".join(out)


# --- one-call helpers ---------------------------------------------------------------------

def hub_svg(spec: HubSpec, theme: str = "Editorial", canvas: str = "landscape", **kw) -> str:
    return scene_to_svg(layout_hub(spec, THEMES[theme], canvas), **kw)


def timeline_svg(spec: TimelineSpec, theme: str = "Editorial", canvas: str = "portrait",
                 style: str = "chronological", **kw) -> str:
    return scene_to_svg(layout_timeline(spec, THEMES[theme], canvas, style), **kw)


def svg_to_png(svg: str, scale: float = 2.0) -> bytes:
    """Rasterise with resvg (self-contained; uses installed system fonts)."""
    import resvg_py
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, zoom=scale))


def export_name(title: str, canvas: str, ext: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:48] or "infographic"
    return f"discover-canada-{slug}-{canvas}.{ext}"


__all__ = ["scene_to_svg", "hub_svg", "timeline_svg", "svg_to_png", "export_name", "CANVASES"]
