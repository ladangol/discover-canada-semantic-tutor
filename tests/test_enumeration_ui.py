"""The scan panel is opt-in and separate from question answering."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def run(monkeypatch, on: bool) -> AppTest:
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    monkeypatch.setenv("STUDY_ENUMERATION", "1" if on else "0")
    for name in [m for m in sys.modules if m == "ui" or m.startswith("ui.")]:
        del sys.modules[name]
    return AppTest.from_file(str(ROOT / "study_app.py"), default_timeout=180).run()


def labels(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def test_the_scan_panel_is_hidden_by_default(monkeypatch):
    at = run(monkeypatch, on=False)
    assert not at.exception and not any("Scan the whole guide" in l for l in labels(at))


def test_the_scan_panel_shows_its_plan_and_cost_before_anything_runs(monkeypatch):
    at = run(monkeypatch, on=True)
    assert not at.exception and any("Scan the whole guide" in l for l in labels(at))
    at.text_input(key="enum_kind").set_value("languages").run()
    shown = " ".join(i.value for i in at.info)
    assert "Plan: read" in shown and "3 AI calls" in shown and "no synonyms" in shown      # cost and scope are stated first
    assert not at.exception
