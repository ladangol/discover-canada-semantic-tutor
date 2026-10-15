"""Design system for infographics: typography scale, semantic colours, spacing, canvases.

Every layout and the SVG renderer read from a `Theme`; nothing is styled per concept.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FONT_STACK = "Inter, 'Helvetica Neue', Helvetica, 'Segoe UI', Arial, sans-serif"


@dataclass(frozen=True)
class Typography:
    title: float = 30          # export title
    subtitle: float = 14.5     # one-to-two-sentence definition under the title
    center: float = 24         # main concept (bold)
    node: float = 17           # related concept (semibold)
    detail: float = 13         # event description
    relation: float = 12.5     # relationship / date chips
    source: float = 11.5       # source attribution
    line_height: float = 1.32


@dataclass(frozen=True)
class Spacing:
    pad: float = 44            # canvas margin
    card_pad_x: float = 16
    card_pad_y: float = 12
    card_radius: float = 12
    gap: float = 10            # minimum clearance between any two boxes


@dataclass(frozen=True)
class CategoryStyle:
    accent: str                # border / bar / connector
    tint: str                  # card fill
    label: str                 # human name, used by the legend


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    ink: str                   # primary text
    muted: str                 # secondary text
    card_text: str
    brand: str                 # centre card and header rule
    on_brand: str
    surface: str               # pills, light fills
    categories: dict[str, CategoryStyle]
    bands: tuple[tuple[str, str], ...]   # (fill, accent) cycle for layered timelines
    type: Typography = field(default_factory=Typography)
    space: Spacing = field(default_factory=Spacing)
    shadow: bool = True


EDITORIAL = Theme(
    name="Editorial",
    bg="#FBF8F3", ink="#1F2430", muted="#6B7280", card_text="#1F2430",
    brand="#C8102E", on_brand="#FFFFFF", surface="#FFFFFF",
    categories={
        "person": CategoryStyle("#2F6DB5", "#E8F0FA", "Person"),
        "organization": CategoryStyle("#7A4FB5", "#F0EAF8", "Organization"),
        "event": CategoryStyle("#D9622B", "#FCEDE4", "Event"),
        "place": CategoryStyle("#2E8B6E", "#E5F3EE", "Place"),
        "concept": CategoryStyle("#B7791F", "#FBF1DC", "Concept"),
    },
    bands=(("#F3ECE1", "#B7791F"), ("#E8F0FA", "#2F6DB5"), ("#E5F3EE", "#2E8B6E"),
           ("#F0EAF8", "#7A4FB5"), ("#FCEDE4", "#D9622B"), ("#EDEFF2", "#5B6472")),
)

MIDNIGHT = Theme(
    name="Midnight",
    bg="#141B2D", ink="#F1F3F8", muted="#A3ACC2", card_text="#F1F3F8",
    brand="#E5384F", on_brand="#FFFFFF", surface="#1E2740",
    categories={
        "person": CategoryStyle("#6AA8F0", "#1F3150", "Person"),
        "organization": CategoryStyle("#B392F0", "#2B2548", "Organization"),
        "event": CategoryStyle("#F0905A", "#3A2A2A", "Event"),
        "place": CategoryStyle("#55C2A0", "#1D3A3A", "Place"),
        "concept": CategoryStyle("#EBB04C", "#3A3226", "Concept"),
    },
    bands=(("#1C2540", "#EBB04C"), ("#1F3150", "#6AA8F0"), ("#1D3A3A", "#55C2A0"),
           ("#2B2548", "#B392F0"), ("#3A2A2A", "#F0905A"), ("#232B3D", "#A3ACC2")),
)

THEMES: dict[str, Theme] = {t.name: t for t in (EDITORIAL, MIDNIGHT)}


@dataclass(frozen=True)
class Canvas:
    name: str
    label: str
    width: int
    height: int | None         # None = grows with content (portrait / tall timelines)


CANVASES: dict[str, Canvas] = {
    "landscape": Canvas("landscape", "Landscape 16:9", 1200, 675),
    "square": Canvas("square", "Square 1:1", 1000, 1000),
    "portrait": Canvas("portrait", "Portrait (mobile / tall)", 560, None),
}
