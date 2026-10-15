"""Geometry for infographics: text measurement, wrapping and collision-free layouts.

Input: a `HubSpec` / `TimelineSpec` + `Theme` + `Canvas`. Output: a `Scene` of plain shapes
(cards, pills, edges, text) in SVG user units. No SVG, no Streamlit here, so layouts can be
unit-tested for overlap and rendered by any backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.infographic_models import HubSpec, TimelineSpec, pages_text
from src.infographic_themes import CANVASES, Canvas, Theme

# =============================================================================
# Scene primitives
# =============================================================================


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float: return self.x + self.w
    @property
    def bottom(self) -> float: return self.y + self.h
    @property
    def cx(self) -> float: return self.x + self.w / 2
    @property
    def cy(self) -> float: return self.y + self.h / 2

    def inflate(self, m: float) -> "Rect":
        return Rect(self.x - m, self.y - m, self.w + 2 * m, self.h + 2 * m)

    def overlaps(self, o: "Rect") -> bool:
        return self.x < o.right and o.x < self.right and self.y < o.bottom and o.y < self.bottom


@dataclass
class Text:
    x: float
    y: float                    # baseline
    text: str
    size: float
    weight: int = 400
    fill: str = "#000"
    anchor: str = "start"       # start | middle | end


@dataclass
class Card:
    id: str
    kind: str                   # center | node | event | chip
    rect: Rect
    fill: str
    stroke: str
    accent: str
    texts: list[Text]
    bar: bool = True            # accent bar on the left edge
    radius: float = 12


@dataclass
class Pill:
    id: str
    rect: Rect
    texts: list[Text]
    fill: str
    stroke: str


@dataclass
class Edge:
    id: str
    d: str                      # SVG path data
    stroke: str
    width: float = 2.0
    opacity: float = 0.75


@dataclass
class Box:                      # bands, stripes
    rect: Rect
    fill: str
    radius: float = 0
    id: str = ""


@dataclass
class Line:
    x1: float; y1: float; x2: float; y2: float
    stroke: str
    width: float = 2.0
    opacity: float = 1.0


@dataclass
class Dot:
    cx: float; cy: float; r: float
    fill: str
    stroke: str = ""


@dataclass
class Scene:
    width: float
    height: float
    theme: Theme
    title: str
    boxes: list[Box] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    pills: list[Pill] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    dots: list[Dot] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)      # header, footer, legend
    overflow: bool = False                               # content did not fit a fixed canvas


# =============================================================================
# Text measurement and wrapping (conservative; the SVG font stack varies by machine)
# =============================================================================

_NARROW = set("iljtfIr.,;:'|!()[]/ ’")
_WIDE = set("mwMW@%")


def _cw(ch: str) -> float:
    if ch in _NARROW:
        return 0.33
    if ch in _WIDE:
        return 0.88
    if ch.isdigit():
        return 0.57
    if ch.isupper():
        return 0.66
    return 0.55


def text_width(text: str, size: float, weight: int = 400) -> float:
    k = 1.07 if weight >= 600 else 1.04 if weight >= 500 else 1.0
    return sum(_cw(c) for c in text) * size * k * 1.04


def wrap(text: str, max_w: float, size: float, weight: int = 400) -> list[str]:
    """Greedy word wrap; over-long words are broken by characters. Never truncates."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if text_width(trial, size, weight) <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
            cur = ""
        while text_width(word, size, weight) > max_w and len(word) > 1:
            k = len(word)
            while k > 1 and text_width(word[:k], size, weight) > max_w:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _on(color: str, theme: Theme) -> str:
    """Readable text colour on a solid accent fill."""
    return "#1A1F2B" if _luminance(color) > 0.55 else "#FFFFFF"


# =============================================================================
# Shared scene furniture: header, footer, legend
# =============================================================================


def _new_scene(theme: Theme, canvas: Canvas, title: str) -> Scene:
    return Scene(width=canvas.width, height=canvas.height or 0, theme=theme, title=title)


def _header(scene: Scene, title: str, subtitle: str) -> float:
    """Adds the title block; returns the y where content may start."""
    t, th, W = scene.theme, scene.theme.type, scene.width
    pad = t.space.pad
    scene.boxes.append(Box(Rect(0, 0, W, 7), t.brand))
    y = 38 + th.title * 0.8
    for line in wrap(title, W - 2 * pad, th.title, 700):
        scene.texts.append(Text(pad, y, line, th.title, 700, t.ink))
        y += th.title * 1.15
    y -= th.title * 0.15
    if subtitle:
        y += 2
        for line in wrap(subtitle, W - 2 * pad, th.subtitle, 400)[:3]:
            y += th.subtitle * 1.35
            scene.texts.append(Text(pad, y, line, th.subtitle, 400, t.muted))
    return y + 22


def _legend_width(cats: list[str], theme: Theme) -> float:
    return sum(14 + text_width(theme.categories[c].label, 11.5) + 18 for c in cats)


def _footer_height(scene_w: float, source: str, cats: list[str], theme: Theme) -> float:
    fit = text_width(source, theme.type.source) + _legend_width(cats, theme) + 40 <= scene_w - 2 * theme.space.pad
    return 46 if fit or not cats else 70


def _footer(scene: Scene, source: str, cats: list[str]) -> None:
    t, th, pad = scene.theme, scene.theme.type, scene.theme.space.pad
    one_line = _footer_height(scene.width, source, cats, t) == 46
    base = scene.height - 18
    scene.lines.append(Line(pad, base - 20, scene.width - pad, base - 20, t.muted, 0.6, 0.35))
    scene.texts.append(Text(pad, base, source, th.source, 400, t.muted))
    x = scene.width - pad - _legend_width(cats, t) if one_line else pad
    y = base if one_line else base - 24
    if not one_line and cats:
        scene.lines.pop()
        scene.lines.append(Line(pad, base - 44, scene.width - pad, base - 44, t.muted, 0.6, 0.35))
        y = base - 20
    for c in cats:
        st = t.categories[c]
        scene.dots.append(Dot(x + 5, y - 4, 5, st.accent))
        scene.texts.append(Text(x + 14, y, st.label, 11.5, 500, t.muted))
        x += 14 + text_width(st.label, 11.5) + 18


def _source_line(pages: list[str]) -> str:
    where = pages_text(pages)
    return "Source: Discover Canada" + (f" · {where}" if where else "")


# =============================================================================
# Cards
# =============================================================================


def _node_card(node_id: str, label: str, small: str, cx: float, cy: float, inner_w: float,
               theme: Theme, category: str, *, size: float | None = None, fixed_w: float | None = None,
               top_left: tuple[float, float] | None = None, min_w: float = 120) -> Card:
    th, sp = theme.type, theme.space
    size = size or th.node
    st = theme.categories[category]
    lines = wrap(label, inner_w, size, 600)
    lh = size * th.line_height
    text_w = max(text_width(l, size, 600) for l in lines)
    w = fixed_w or max(text_w + 2 * sp.card_pad_x + 6, min_w)
    h = 2 * sp.card_pad_y + len(lines) * lh + (th.source + 6 if small else 0)
    rect = Rect(top_left[0], top_left[1], w, h) if top_left else Rect(cx - w / 2, cy - h / 2, w, h)
    x = rect.x + 6 + sp.card_pad_x
    texts = [Text(x, rect.y + sp.card_pad_y + size * 0.95 + i * lh, l, size, 600, theme.card_text)
             for i, l in enumerate(lines)]
    if small:
        texts.append(Text(x, rect.bottom - sp.card_pad_y + 1, small, th.source, 400, theme.muted))
    return Card(node_id, "node", rect, st.tint, st.accent, st.accent, texts, True, sp.card_radius)


def _measure_node(label: str, small: str, inner_w: float, theme: Theme, min_w: float = 120) -> tuple[float, float]:
    c = _node_card("m", label, small, 0, 0, inner_w, theme, "concept", min_w=min_w)
    return c.rect.w, c.rect.h


def _center_card(label: str, category_hint: str, cx: float, cy: float, theme: Theme, inner_w: float,
                 *, left: float | None = None, top: float | None = None, width: float | None = None) -> Card:
    th = theme.type
    size = th.center
    lines = wrap(label, inner_w, size, 700)
    if len(lines) >= 3:                       # long names: step down within the 22-26 px range
        size = max(22.0, th.center - 2)
        lines = wrap(label, inner_w, size, 700)
    lh = size * 1.2
    w = width or max(max(text_width(l, size, 700) for l in lines) + 44, 150)
    h = len(lines) * lh + 30
    rect = Rect(left, top, w, h) if left is not None else Rect(cx - w / 2, cy - h / 2, w, h)
    anchor_x = rect.cx if left is None or width is None else rect.x + 22
    anchor = "middle" if left is None or width is None else "start"
    if left is not None and width is not None:
        anchor_x, anchor = rect.x + 22, "start"
    texts = [Text(anchor_x, rect.y + 15 + size * 0.92 + i * lh, l, size, 700, theme.on_brand, anchor)
             for i, l in enumerate(lines)]
    return Card("center", "center", rect, theme.brand, theme.brand, theme.brand, texts, False, 14)


def _edge_point(r: Rect, tx: float, ty: float, margin: float = 2) -> tuple[float, float]:
    """Where the ray from r's centre towards (tx, ty) leaves the rectangle."""
    dx, dy = tx - r.cx, ty - r.cy
    if dx == 0 and dy == 0:
        return r.cx, r.cy
    sx = (r.w / 2 + margin) / abs(dx) if dx else math.inf
    sy = (r.h / 2 + margin) / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return r.cx + dx * s, r.cy + dy * s


PILL_MAX_W = 200.0


def _pill(pid: str, text: str, cx: float, cy: float, theme: Theme, accent: str, max_w: float = PILL_MAX_W) -> Pill:
    size = theme.type.relation
    lines = wrap(text, max_w, size, 500)
    w = max(text_width(l, size, 500) for l in lines) + 20
    lh = size * 1.25
    h = len(lines) * lh + 8
    rect = Rect(cx - w / 2, cy - h / 2, w, h)
    texts = [Text(rect.cx, rect.y + 4 + size * 0.92 + i * lh, l, size, 500, theme.card_text, "middle")
             for i, l in enumerate(lines)]
    return Pill(pid, rect, texts, theme.surface, accent)


def _place_pill(pid: str, text: str, points: list[tuple[float, float]], theme: Theme, accent: str,
                obstacles: list[Rect], bounds: Rect | None = None) -> Pill:
    """Nearest collision-free position to the connector midpoint: first on the line, then just beside it."""
    mid = points[0]
    cands: list[tuple[float, float]] = list(points)
    if len(points) > 1:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        # direction of the connector, estimated from the sampled points
        xa, ya = points[3] if len(points) > 3 else points[0]
        xb, yb = points[4] if len(points) > 4 else points[-1]
        dx, dy = (xb - xa), (yb - ya)
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        ph = _pill(pid, text, 0, 0, theme, accent).rect.h
        for t in range(len(points)):
            bx, by = points[t]
            for off in (ph / 2 + 10, -(ph / 2 + 10), ph / 2 + 34, -(ph / 2 + 34)):
                cands.append((bx + nx * off, by + ny * off))
    cands.sort(key=lambda c: math.hypot(c[0] - mid[0], c[1] - mid[1]) + (0 if c in points else 25))
    best = None
    for max_w in (PILL_MAX_W, 150.0, 115.0):          # wrap narrower before giving up
        for cx, cy in cands:
            p = _pill(pid, text, cx, cy, theme, accent, max_w=max_w)
            inside = bounds is None or (p.rect.x >= bounds.x and p.rect.right <= bounds.right
                                        and p.rect.y >= bounds.y and p.rect.bottom <= bounds.bottom)
            if inside and not any(p.rect.inflate(3).overlaps(o) for o in obstacles):
                return p
            best = best or p
    return best  # type: ignore[return-value]


# =============================================================================
# Concept hub layouts
# =============================================================================


def _overlap_free(rects: list[Rect], gap: float) -> bool:
    return not any(rects[i].inflate(gap / 2).overlaps(rects[j].inflate(gap / 2))
                   for i in range(len(rects)) for j in range(i + 1, len(rects)))


def layout_hub(spec: HubSpec, theme: Theme, canvas: Canvas | str) -> Scene:
    canvas = CANVASES[canvas] if isinstance(canvas, str) else canvas
    scene = _new_scene(theme, canvas, spec.title)
    cats = sorted({n.category for n in spec.nodes} | {spec.center_category},
                  key=lambda c: list(theme.categories).index(c))
    source = _source_line(spec.pages)
    top = _header(scene, spec.title, spec.subtitle)
    foot = _footer_height(scene.width, source, cats, theme)
    pad = theme.space.pad
    n = len(spec.nodes)

    if canvas.name == "portrait":
        bottom = _hub_portrait(scene, spec, top)
        scene.height = bottom + foot + 8
    else:
        content = Rect(pad, top, scene.width - 2 * pad, scene.height - top - foot)
        (_hub_compact if n <= 3 else _hub_radial)(scene, spec, content)
    _footer(scene, source, cats)
    return scene


def _small(node) -> str:
    return pages_text(node.pages)


def _radial_attempt(spec: HubSpec, content: Rect, th: Theme, node_inner: float, center_inner: float,
                    side_factor: float):
    """Build one candidate radial layout; returns (center, cards, edges, pills, collision_count)."""
    n = len(spec.nodes)
    angles = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    side = [abs(math.cos(a)) > 0.7 for a in angles]
    center = _center_card(spec.title, spec.center_category, content.cx, content.cy, th, center_inner)
    inners = [node_inner * (side_factor if sd else 1.0) for sd in side]
    sizes = [_measure_node(nd.label, _small(nd), w, th) for nd, w in zip(spec.nodes, inners)]
    mw, mh = max(s[0] for s in sizes), max(s[1] for s in sizes)
    rx, ry = content.w / 2 - mw / 2, content.h / 2 - mh / 2
    cards = [_node_card(nd.id, nd.label, _small(nd), content.cx + rx * math.cos(a), content.cy + ry * math.sin(a),
                        w, th, nd.category)
             for nd, a, w in zip(spec.nodes, angles, inners)]
    obstacles = [center.rect] + [c.rect for c in cards]
    edges: list[Edge] = []
    pills: list[Pill] = []
    for nd, card in zip(spec.nodes, cards):
        a_ = _edge_point(center.rect, card.rect.cx, card.rect.cy)
        b_ = _edge_point(card.rect, center.rect.cx, center.rect.cy)
        acc = th.categories[nd.category].accent
        edges.append(Edge(nd.id, f"M{a_[0]:.1f},{a_[1]:.1f} L{b_[0]:.1f},{b_[1]:.1f}", acc))
        pts = [(a_[0] + (b_[0] - a_[0]) * t, a_[1] + (b_[1] - a_[1]) * t) for t in (0.5, 0.4, 0.6, 0.3, 0.7)]
        pills.append(_place_pill(nd.id, nd.relationship, pts, th, acc, obstacles + [p.rect for p in pills], content))
    plain = [center.rect] + [c.rect for c in cards] + [p.rect for p in pills]
    boxes = [center.rect.inflate(th.space.gap / 2)] + [c.rect.inflate(th.space.gap / 2) for c in cards] \
        + [p.rect for p in pills]
    clashes = sum(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    clashes += sum(not (content.x - 1 <= r.x and r.right <= content.right + 1 and content.y - 1 <= r.y
                        and r.bottom <= content.bottom + 1) for r in plain)
    return center, cards, edges, pills, clashes


def _hub_radial(scene: Scene, spec: HubSpec, content: Rect) -> None:
    """Cards on an ellipse around the centre. Tries card widths from roomy to tight and keeps the first
    configuration with no overlapping cards/pills; otherwise the one with the fewest clashes."""
    th = scene.theme
    best = None
    for node_inner in (235.0, 215.0, 195.0, 175.0, 155.0):
        for center_inner in (300.0, 260.0, 220.0, 185.0):
            for side_factor in (1.0, 0.8, 0.65):
                got = _radial_attempt(spec, content, th, node_inner, center_inner, side_factor)
                if best is None or got[4] < best[4]:
                    best = got
                if got[4] == 0:
                    break
            if best[4] == 0:
                break
        if best[4] == 0:
            break
    center, cards, edges, pills, clashes = best
    scene.cards.append(center)
    scene.cards += cards
    scene.edges += edges
    scene.pills += pills
    if clashes:
        scene.overflow = True      # could not fit cleanly; the UI suggests the portrait layout


def _hub_compact(scene: Scene, spec: HubSpec, content: Rect) -> None:
    """One to three relationships: a tidy fan under the centre card (no invented nodes)."""
    th = scene.theme
    n = len(spec.nodes)
    center = _center_card(spec.title, spec.center_category, content.cx, 0, th, 300)
    inner = min(270.0, content.w / max(n, 1) - 70)
    min_w = min(250.0, content.w / max(n, 1) - 30)
    sizes = [_measure_node(nd.label, _small(nd), inner, th, min_w) for nd in spec.nodes] or [(0, 0)]
    row_h = max(s[1] for s in sizes)
    vgap = min(max(content.h - center.rect.h - row_h, 90), 150)
    group_h = center.rect.h + vgap + (row_h if n else 0)
    y0 = content.y + max((content.h - group_h) / 2, 0)
    dy = y0 - center.rect.y
    center.rect.y += dy
    for t in center.texts:
        t.y += dy
    scene.cards.append(center)
    row_y = y0 + center.rect.h + vgap
    obstacles = [center.rect]
    for i, nd in enumerate(spec.nodes):
        cx = content.x + content.w * (i + 0.5) / n
        card = _node_card(nd.id, nd.label, _small(nd), cx, 0, inner, th, nd.category,
                          top_left=(cx - sizes[i][0] / 2, row_y), min_w=min_w)
        scene.cards.append(card)
        obstacles.append(card.rect)
    for i, (nd, card) in enumerate(zip(spec.nodes, scene.cards[1:])):
        spread = min(70.0, center.rect.w / max(n, 1))
        sx = center.rect.cx + (i - (n - 1) / 2) * spread
        sy, ex, ey = center.rect.bottom, card.rect.cx, card.rect.y
        mid = (sy + ey) / 2
        acc = th.categories[nd.category].accent
        scene.edges.append(Edge(nd.id, f"M{sx:.1f},{sy:.1f} C{sx:.1f},{mid:.1f} {ex:.1f},{mid:.1f} {ex:.1f},{ey:.1f}", acc))
        pts = [(sx * (1 - t) ** 3 + 3 * sx * (1 - t) ** 2 * t + 3 * ex * (1 - t) * t ** 2 + ex * t ** 3,
                sy * (1 - t) ** 3 + 3 * mid * (1 - t) ** 2 * t + 3 * mid * (1 - t) * t ** 2 + ey * t ** 3)
               for t in (0.5, 0.4, 0.6, 0.3, 0.7)]
        scene.pills.append(_place_pill(nd.id, nd.relationship, pts, th, acc, obstacles + [p.rect for p in scene.pills], content))


def _hub_portrait(scene: Scene, spec: HubSpec, top: float) -> float:
    """Single column for phones: centre card, then each relationship as a labelled card on a spine."""
    th, W, pad = scene.theme, scene.width, scene.theme.space.pad
    spine_x = pad + 8
    center = _center_card(spec.title, spec.center_category, 0, 0, th, W - 2 * pad - 44,
                          left=pad, top=top, width=W - 2 * pad)
    scene.cards.append(center)
    y = center.rect.bottom
    card_x, card_w = pad + 34, W - 2 * pad - 34
    first_y = y
    for nd in spec.nodes:
        acc = th.categories[nd.category].accent
        y += 14
        probe = _pill(nd.id, nd.relationship, 0, 0, th, acc, max_w=card_w - 30)
        pill = _pill(nd.id, nd.relationship, card_x + probe.rect.w / 2, y + probe.rect.h / 2, th, acc, max_w=card_w - 30)
        scene.pills.append(pill)
        y = pill.rect.bottom + 6
        card = _node_card(nd.id, nd.label, _small(nd), 0, 0, card_w - 6 - 32, th, nd.category,
                          fixed_w=card_w, top_left=(card_x, y))
        scene.cards.append(card)
        scene.edges.append(Edge(nd.id, f"M{spine_x:.1f},{card.rect.cy:.1f} L{card_x:.1f},{card.rect.cy:.1f}", acc, 2.0))
        scene.dots.append(Dot(spine_x, card.rect.cy, 5, acc, th.bg))
        y = card.rect.bottom
    if spec.nodes:
        scene.lines.append(Line(spine_x, first_y, spine_x, y - scene.cards[-1].rect.h / 2, th.muted, 2, 0.45))
    return y + 16


# =============================================================================
# Timeline layouts
# =============================================================================


def max_events(canvas: Canvas | str, style: str) -> int | None:
    """How many events fit a fixed canvas (None = unlimited, the canvas grows)."""
    canvas = CANVASES[canvas] if isinstance(canvas, str) else canvas
    if canvas.name == "portrait" or style == "layered":
        return None
    card_w, pad = 210, 44
    return int((canvas.width - 2 * pad - card_w) // (card_w / 2 + 12)) + 1


def layout_timeline(spec: TimelineSpec, theme: Theme, canvas: Canvas | str, style: str = "chronological") -> Scene:
    canvas = CANVASES[canvas] if isinstance(canvas, str) else canvas
    scene = _new_scene(theme, canvas, spec.title)
    top = _header(scene, spec.title, spec.subtitle)
    source = _source_line(spec.pages)
    foot = _footer_height(scene.width, source, [], theme)
    pad = theme.space.pad
    bands = list(dict.fromkeys(e.band for e in spec.events))
    band_of = {b: i for i, b in enumerate(bands)}

    if style == "layered":
        end = _timeline_layered(scene, spec, top, band_of)
    elif canvas.name == "portrait":
        end = _timeline_vertical(scene, spec, top, band_of)
    else:
        cap = max_events(canvas, style) or len(spec.events)
        spec = TimelineSpec(spec.title, spec.subtitle, spec.events[:cap], spec.pages)
        end = _timeline_horizontal(scene, spec, Rect(pad, top, scene.width - 2 * pad, scene.height - top - foot), band_of)
    needed = end + foot
    if canvas.height is None:
        scene.height = needed
    elif needed > scene.height:
        scene.height, scene.overflow = needed, True
    _footer(scene, source, [])
    return scene


def _event_card(e, x: float, y: float, w: float, theme: Theme, accent: str, *, big: bool = True,
                top_left: bool = True, anchor_bottom: float | None = None) -> Card:
    th, sp = theme.type, theme.space
    inner = w - 6 - 2 * sp.card_pad_x
    t_size, d_size = (th.node - 1, th.detail) if big else (th.node - 3, th.detail - 1)
    date_lines = wrap(e.date_label, inner, th.relation + 0.5, 700)
    title_lines = wrap(e.title, inner, t_size, 600)
    desc_lines = wrap(e.description, inner, d_size, 400) if (big and e.description) else []
    src = pages_text(e.pages)
    h = (2 * sp.card_pad_y + len(date_lines) * (th.relation + 0.5) * 1.3 + 2 + len(title_lines) * t_size * 1.3
         + (6 + len(desc_lines) * d_size * 1.4 if desc_lines else 0) + (th.source + 8 if big and src else 0))
    if anchor_bottom is not None:
        y = anchor_bottom - h
    rect = Rect(x, y, w, h)
    tx = x + 6 + sp.card_pad_x
    cy = y + sp.card_pad_y
    texts: list[Text] = []
    for l in date_lines:
        cy += (th.relation + 0.5) * 1.3
        texts.append(Text(tx, cy - 3, l, th.relation + 0.5, 700, accent))
    cy += 2
    for l in title_lines:
        cy += t_size * 1.3
        texts.append(Text(tx, cy - 4, l, t_size, 600, theme.card_text))
    if desc_lines:
        cy += 6
        for l in desc_lines:
            cy += d_size * 1.4
            texts.append(Text(tx, cy - 4, l, d_size, 400, theme.muted))
    if big and src:
        texts.append(Text(tx, rect.bottom - sp.card_pad_y + 1, src, th.source, 400, theme.muted))
    return Card(e.id, "event", rect, theme.surface, accent, accent, texts, True, sp.card_radius)


def _timeline_vertical(scene: Scene, spec: TimelineSpec, top: float, band_of: dict[str, int]) -> float:
    th, W, pad = scene.theme, scene.width, scene.theme.space.pad
    spine_x, card_x = pad + 8, pad + 34
    y = top + 4
    y_first = y
    for e in spec.events:
        acc = th.bands[band_of[e.band] % len(th.bands)][1]
        card = _event_card(e, card_x, y, W - pad - card_x, th, acc)
        scene.cards.append(card)
        scene.edges.append(Edge(e.id, f"M{spine_x:.1f},{card.rect.y + 20:.1f} L{card_x:.1f},{card.rect.y + 20:.1f}", acc, 2.0))
        scene.dots.append(Dot(spine_x, card.rect.y + 20, 6, acc, th.bg))
        y = card.rect.bottom + 14
    if spec.events:
        scene.lines.append(Line(spine_x, y_first, spine_x, scene.cards[-1].rect.y + 20, th.muted, 2, 0.45))
    return y


def _timeline_horizontal(scene: Scene, spec: TimelineSpec, content: Rect, band_of: dict[str, int]) -> float:
    th = scene.theme
    ev = spec.events
    n = len(ev)
    axis_y = content.cy
    if n == 0:
        return content.bottom
    # fewer events -> wider cards (shorter text blocks); same-side cards stay >= 12 apart
    cw = min(300.0, max(210.0, (content.w - 12 * (n - 1)) / (1 + (n - 1) / 2))) if n > 1 else 300.0
    step = (content.w - cw) / (n - 1) if n > 1 else 0
    chip_max = max(min(step - 18, 130.0), 70.0) if n > 1 else 130.0
    scene.lines.append(Line(content.x, axis_y, content.right, axis_y, th.muted, 3, 0.5))
    for i, e in enumerate(ev):
        cx = content.x + cw / 2 + i * step if n > 1 else content.cx
        acc = th.bands[band_of[e.band] % len(th.bands)][1]
        above = i % 2 == 0
        chip = _pill(e.id, e.date_label, cx, axis_y, th, acc, max_w=chip_max)
        chip.texts = [Text(t.x, t.y, t.text, t.size, 700, th.card_text, "middle") for t in chip.texts]
        gap = 30 + chip.rect.h / 2
        if above:
            card = _event_card(e, cx - cw / 2, 0, cw, th, acc, anchor_bottom=axis_y - gap)
        else:
            card = _event_card(e, cx - cw / 2, axis_y + gap, cw, th, acc)
        stem_y = card.rect.bottom if above else card.rect.y
        scene.lines.append(Line(cx, stem_y, cx, axis_y + (-chip.rect.h / 2 if above else chip.rect.h / 2), acc, 2, 0.8))
        scene.cards.append(card)
        scene.pills.append(chip)
        scene.edges.append(Edge(e.id, f"M{cx:.1f},{axis_y:.1f} L{cx:.1f},{stem_y:.1f}", acc, 0.01, 0.0))
    lo = min((c.rect.y for c in scene.cards), default=content.y)
    hi = max((c.rect.bottom for c in scene.cards), default=content.bottom)
    if lo < content.y or hi > content.bottom:
        scene.overflow = True
    return max(hi, content.bottom)


def _timeline_layered(scene: Scene, spec: TimelineSpec, top: float, band_of: dict[str, int]) -> float:
    th, W, pad = scene.theme, scene.width, scene.theme.space.pad
    portrait = W < 700
    label_w = 0 if portrait else (170 if W > 1000 else 150)
    chip_w = 250.0 if portrait else (196.0 if W > 1000 else 190.0)
    gap = 10
    area_x = pad + (label_w + 14 if not portrait else 14)
    area_w = W - pad - area_x - 14
    cols = max(1, int((area_w + gap) // (chip_w + gap)))
    chip_w = (area_w - gap * (cols - 1)) / cols
    y = top + 2
    groups: dict[str, list] = {}
    for e in spec.events:
        groups.setdefault(e.band, []).append(e)
    for b, evs in groups.items():
        fill, acc = th.bands[band_of[b] % len(th.bands)]
        years = [yr for e in evs for yr in (e.start_year, e.end_year) if yr]
        span = (f"{min(years)}–{max(years)}" if years and min(years) != max(years) else (str(years[0]) if years else ""))
        # lay chips out first to know the band height
        head_h = 0.0
        label_lines = wrap(b, (label_w - 28) if not portrait else (W - 2 * pad - 28), th.type.node - 2, 700)
        if portrait:
            head_h = len(label_lines) * 22 + (18 if span else 0) + 14
        rows: list[list[Card]] = []
        cards = [_event_card(e, 0, 0, chip_w, th, acc, big=False) for e in evs]
        for i in range(0, len(cards), cols):
            rows.append(cards[i:i + cols])
        body_h = sum(max(c.rect.h for c in r) for r in rows) + gap * (len(rows) - 1)
        label_h = len(label_lines) * 22 + (18 if span else 0) + 28
        band_h = max(head_h + body_h + 24, label_h if not portrait else 0)
        scene.boxes.append(Box(Rect(pad, y, W - 2 * pad, band_h), fill, 14))
        if not portrait:
            scene.boxes.append(Box(Rect(pad, y, label_w, band_h), acc, 14))
            lc = _on(acc, th)
            ty = y + 14 + (th.type.node - 2) * 0.95
            for l in label_lines:
                scene.texts.append(Text(pad + 14, ty, l, th.type.node - 2, 700, lc))
                ty += 22
            if span:
                scene.texts.append(Text(pad + 14, ty - 2, span, th.type.relation, 500, lc))
        else:
            ty = y + 12 + (th.type.node - 2) * 0.95
            for l in label_lines:
                scene.texts.append(Text(pad + 14, ty, l, th.type.node - 2, 700, th.card_text))
                ty += 22
            if span:
                scene.texts.append(Text(pad + 14, ty - 2, span, th.type.relation, 500, acc))
        ry = y + (head_h + 12 if portrait else 12)
        for r in rows:
            rh = max(c.rect.h for c in r)
            for k, c in enumerate(r):
                dx, dy = area_x + k * (chip_w + gap) - c.rect.x, ry - c.rect.y
                c.rect.x += dx
                c.rect.y += dy
                for t in c.texts:
                    t.x += dx
                    t.y += dy
                c.rect.h = rh
                scene.cards.append(c)
            ry += rh + gap
        y += band_h + 12
    return y


def fit_count(spec: TimelineSpec, theme: Theme, canvas: Canvas | str, style: str) -> int:
    """Largest number of leading events that fit the canvas without growing it."""
    canvas = CANVASES[canvas] if isinstance(canvas, str) else canvas
    if canvas.height is None or not spec.events:
        return len(spec.events)
    cap = min(len(spec.events), max_events(canvas, style) or len(spec.events))
    for n in range(cap, 0, -1):
        sub = TimelineSpec(spec.title, spec.subtitle, spec.events[:n], spec.pages)
        if not layout_timeline(sub, theme, canvas, style).overflow:
            return n
    return 1
