"""The Talk page is opt-in and renders cleanly; the other pages are unaffected. (No models or API key needed.)"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def run(monkeypatch, voice: bool, page: str | None = None) -> AppTest:
    # each AppTest has its own component registry; re-import ui.* so their v2 components register again
    for name in [m for m in sys.modules if m == "ui" or m.startswith("ui.")]:
        del sys.modules[name]
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    monkeypatch.delenv("TTS_VOICE_PATH", raising=False)
    monkeypatch.setenv("STUDY_VOICE", "1" if voice else "0")
    at = AppTest.from_file(str(ROOT / "study_app.py"), default_timeout=180)
    if page:
        at.session_state["page"] = page
    return at.run()


def nav_keys(at: AppTest) -> list[str]:
    return [b.key for b in at.button if b.key and b.key.startswith("nav_")]


def test_talk_is_hidden_by_default_and_cannot_be_reached(monkeypatch):
    at = run(monkeypatch, voice=False, page="Talk")
    assert not at.exception
    assert not any("talk" in k for k in nav_keys(at))
    assert any(b.key == "nav_learn_on" for b in at.button)           # falls back to Learn


def test_talk_page_renders_when_enabled_even_without_models(monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "piper")                          # Piper without a voice file = "not set up"
    at = run(monkeypatch, voice=True, page="Talk")
    assert not at.exception
    assert any("talk" in k for k in nav_keys(at))
    page_text = " ".join(str(getattr(i, "value", "")) for i in list(at.info) + list(at.markdown))
    assert "Setup needed" in page_text                                # no Whisper/Piper configured: clear message, no crash


def test_other_pages_still_render_with_voice_enabled(monkeypatch):
    for page in ("Learn", "Practice", "Flashcards"):
        assert not run(monkeypatch, voice=True, page=page).exception


def run_atlas(monkeypatch, voice: bool) -> AppTest:
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    monkeypatch.delenv("TTS_VOICE_PATH", raising=False)
    monkeypatch.setenv("STUDY_VOICE", "1" if voice else "0")
    for name in [m for m in sys.modules if m == "ui" or m.startswith("ui.") or m == "src.atlas3d"]:
        del sys.modules[name]
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def test_atlas_gets_a_voice_tab_only_when_enabled(monkeypatch):
    off, on = run_atlas(monkeypatch, voice=False), run_atlas(monkeypatch, voice=True)
    assert not off.exception and not on.exception
    assert not any("Voice" in t.label for t in off.tabs)
    assert any("Voice" in t.label for t in on.tabs)
