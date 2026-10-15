"""Keep tests from touching the real usage ledger or reading a real key."""
import pytest


@pytest.fixture(autouse=True)
def isolated_usage_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path / "data"))
    for name in ("STUDY_DAILY_USD_BUDGET", "STUDY_DAILY_TOKEN_BUDGET", "VENICE_PRICE_IN_PER_M", "VENICE_PRICE_OUT_PER_M"):
        monkeypatch.delenv(name, raising=False)
