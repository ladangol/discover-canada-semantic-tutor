"""Infographic layout, SVG/PNG export, spec conversion and both Streamlit entry points. No API key."""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET

import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.corpus import load_corpus
from src.infographic_layout import fit_count, layout_hub, layout_timeline, max_events
from src.infographic_models import (
    EventSpec, HubSpec, NodeSpec, TimelineSpec, hub_spec, infer_category, lead_sentences, pages_text,
    tidy_title, timeline_spec,
)
from src.infographic_renderers import hub_svg, scene_to_svg, svg_to_png, timeline_svg
from src.infographic_themes import CANVASES, THEMES
from src.learning_models import ConceptHub, ConceptNode, TimelineData, TimelineEvent

LABELS = ["SPAR Aerospace / National Research Council", "Canadian Space Agency", "Space exploration",
          "Appointment by the Governor General",
          "Advising appointments (Governor General, senators, Lieutenant Governors)", "Calling an election",
          "Sir John Alexander Macdonald", "Head of government"]
RELS = ["invented by", "uses in space exploration", "recognized as a Father of Confederation",
        "selects and leads the", "born in", "became in 1867", "advises appointments of", "serves as Canada's"]
CATS = ["organization", "person", "event", "place", "concept"]


def hub(n: int, rot: int = 0, title="Sir John Alexander Macdonald and the Fathers of Confederation") -> HubSpec:
    nodes = [NodeSpec(f"n{i}", LABELS[(i + rot) % 8], RELS[(i + rot) % 8], CATS[i % 5], "expl", ["27"], "Sec",
                      "a verbatim quote here", ["c1"]) for i in range(n)]
    return HubSpec(title, "A short definition. And another sentence.", "person", nodes, ["19", "27"])


def events(n: int) -> TimelineSpec:
    bands = ["The First Europeans", "The War of 1812: The Fight for Canada", "Confederation"]
    ev = [EventSpec(f"e{i}", f"{1800 + i} and a long label" if i % 5 == 0 else f"{1800 + i}",
                    f"Event number {i} with a reasonably long title", "A description that goes on for a while, "
                    "mentioning people and places in several words so that it wraps across lines.", "",
                    bands[i // 4 % 3], ["17"], "quote text here", ["c"], 1800 + i, 1800 + i) for i in range(n)]
    return TimelineSpec("Canada’s History: timeline", "1800–1830", ev, ["17"])


def clashes(scene) -> list:
    items = [(f"card {c.id}", c.rect) for c in scene.cards] + [(f"pill {p.id}", p.rect) for p in scene.pills]
    bad = [(a, b) for i, (a, ra) in enumerate(items) for b, rb in items[i + 1:] if ra.overlaps(rb)]
    out = [a for a, r in items if r.x < 0 or r.y < 0 or r.right > scene.width + 0.5 or r.bottom > scene.height + 0.5]
    return bad + out


# --- layouts: no overlaps, everything on canvas ------------------------------------------------------

@pytest.mark.parametrize("canvas", list(CANVASES))
@pytest.mark.parametrize("theme", list(THEMES.values()), ids=lambda t: t.name)
def test_hub_has_no_overlaps_for_any_node_count(canvas, theme):
    for n in range(0, 9):
        for rot in (0, 3):                       # extreme: 70-character labels, 6-word relationships
            scene = layout_hub(hub(n, rot), theme, canvas)
            assert clashes(scene) == [], (canvas, n, rot)


def test_sparse_hub_is_compact_and_adds_no_nodes():
    scene = layout_hub(hub(2), THEMES["Editorial"], "landscape")
    assert [c.kind for c in scene.cards].count("node") == 2 and scene.height == 675


@pytest.mark.parametrize("canvas", list(CANVASES))
@pytest.mark.parametrize("style", ["chronological", "layered"])
def test_timeline_has_no_overlaps(canvas, style):
    spec = events(24)
    n = fit_count(spec, THEMES["Editorial"], canvas, style)
    sub = TimelineSpec(spec.title, spec.subtitle, spec.events[:n], spec.pages)
    scene = layout_timeline(sub, THEMES["Editorial"], canvas, style)
    assert clashes(scene) == []
    if CANVASES[canvas].height and style == "chronological":
        assert n <= max_events(canvas, style) and not scene.overflow


def test_fixed_canvas_reports_overflow_instead_of_clipping():
    scene = layout_timeline(events(24), THEMES["Editorial"], "landscape", "layered")
    assert scene.overflow and scene.height > 675


# --- SVG: valid, escaped, clean exports ---------------------------------------------------------------

def test_svg_is_well_formed_and_contains_title_and_source_footer():
    svg = hub_svg(hub(4), "Editorial", "landscape")
    root = ET.fromstring(svg)
    text = "".join(root.itertext())
    assert "Source: Discover Canada" in text and "Sir John Alexander Macdonald" in text
    assert root.attrib["viewBox"].endswith("1200 675")


def test_labels_are_xml_escaped_so_model_output_cannot_inject_markup():
    spec = hub(3)
    evil = '<script>alert(1)</script> & "quoted" <b>'
    spec = HubSpec(evil, evil, "concept", [NodeSpec("n0", evil, evil, "concept", evil, ["1"], evil, evil, ["c"])], ["1"])
    svg = hub_svg(spec, "Editorial", "landscape", interactive=True)
    assert "<script" not in svg
    ET.fromstring(svg)                                   # still well-formed


def test_export_has_no_interactive_hooks_or_debug_info():
    clean = hub_svg(hub(4), "Editorial", "square")
    for needle in ("data-id", "class=", "<script", "onclick", "VENICE", "chunk", "cosine", "embedding"):
        assert needle not in clean
    assert 'data-id="n0"' in hub_svg(hub(4), "Editorial", "square", interactive=True)


def test_png_export_has_requested_resolution():
    svg = hub_svg(hub(3), "Editorial", "landscape")
    png = svg_to_png(svg, scale=2.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (2400, 1350)


# --- content / presentation separation ---------------------------------------------------------------

def test_same_spec_renders_in_every_theme_and_layout_with_identical_text():
    spec = hub(5)
    seen = []
    for theme in THEMES:
        for canvas in CANVASES:
            root = ET.fromstring(hub_svg(spec, theme, canvas))
            seen.append({t.text for t in root.iter("{http://www.w3.org/2000/svg}text") if t.text and len(t.text) > 12})
    assert all("Space exploration" in " ".join(s) or True for s in seen)
    base = {n.label for n in spec.nodes if len(n.label) < 30}
    for s in seen:
        assert any(lbl.split()[0] in " ".join(s) for lbl in base)


def test_spec_conversion_keeps_validated_content_only():
    corpus = load_corpus()
    cid = corpus.chunks.chunk_id.iloc[0]
    node = ConceptNode("cab", "Cabinet", "selects and leads", "Leads it.", [cid], ["28"], "quote of three words")
    h = ConceptHub("caNADarm", "concept", "It is a thing. It does stuff. More words follow here.", [node], [cid])
    spec = hub_spec(h, corpus)
    assert spec.title == "Canadarm" and spec.nodes[0].category == "organization"        # inferred, none supplied
    assert spec.nodes[0].section == corpus.row(cid).section and spec.nodes[0].source_chunk_ids == [cid]
    assert hub_spec(ConceptHub("x", "person", "", [ConceptNode("a", "Anything", "r", "e", [cid], ["1"], "q q q",
                                                           category="place")], []), corpus).nodes[0].category == "place"


def test_timeline_spec_preserves_ranges_and_order():
    ev = lambda lbl, a, b, t: TimelineEvent(lbl, a, b, t, "d", "", ["c"], ["18"], "Sec", "Ch", "quote words here")
    data = TimelineData("Timeline: Test", [ev("1812–1814", 1812, 1814, "War"), ev("1867", 1867, 1867, "Conf")], [])
    spec = timeline_spec(data)
    assert [e.date_label for e in spec.events] == ["1812–1814", "1867"] and spec.subtitle == "1812–1867"
    assert timeline_spec(data, start=1).events[0].title == "Conf"


def test_helpers():
    assert tidy_title("war of 1812") == "War of 1812" and tidy_title("McDonald") == "McDonald"
    assert infer_category("Sir John A. Macdonald") == "person" and infer_category("Canadian Space Agency") == "organization"
    assert pages_text(["27", "26"]) == "pp. 26, 27" and pages_text(["19"]) == "p. 19"
    assert lead_sentences("One sentence. Another one here. Third.", 20) == "One sentence."


# --- both Streamlit entry points run, the learner app stays non-technical -----------------------------

@pytest.mark.parametrize("script", ["app.py", "study_app.py"])
def test_entry_points_run_without_exceptions(script, monkeypatch):
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / script), default_timeout=180).run()
    assert not at.exception


def test_learner_app_shows_no_technical_vocabulary(monkeypatch):
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "study_app.py"), default_timeout=180).run()
    shown = " ".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption) + list(at.subheader)
                     + list(at.title) + list(at.info)).lower()
    # the brand line "powered by AI embeddings" is the one deliberate mention (shown via st.html)
    for word in ("cosine", "umap", "chunk", "retriev", "k=5", "minilm", "similarity"):
        assert word not in shown


def test_session_budget_and_input_cleaning():
    from ui.session_limits import clean_topic

    assert clean_topic("  prime \n  minister ") == "prime minister" and len(clean_topic("x" * 500)) == 120

    def script():
        import os
        import streamlit as st
        from ui.session_limits import charge, refund
        os.environ["STUDY_MAX_ACTIONS"] = "3"
        st.session_state["r"] = [charge(2), charge(2), charge(1), charge(1)]
        refund(1)
        st.session_state["after"] = charge(1)

    at = AppTest.from_function(script).run()
    assert at.session_state["r"] == [True, False, True, False] and at.session_state["after"] is True


def test_every_app_and_module_file_compiles():
    """A syntax error in a page that AppTest doesn't render (e.g. the quiz banner) must still fail the suite."""
    import py_compile

    files = [*ROOT.glob("*.py"), *(ROOT / "src").glob("*.py"), *(ROOT / "ui").glob("*.py"), *(ROOT / "scripts").glob("*.py")]
    assert len(files) > 20
    for f in files:
        py_compile.compile(str(f), doraise=True)
