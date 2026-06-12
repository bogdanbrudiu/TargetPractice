from pathlib import Path


def test_auto_record_debounce_is_100ms():
    # Keep a quick regression guard for the expected debounce.
    # This is intentionally a source-level assertion (the loop is time-based).
    base = Path(__file__).resolve().parent.parent
    src = (base / "targetweb" / "server.py").read_text(encoding="utf-8")
    assert "RECORD_DEBOUNCE_SECONDS = 0.10" in src
    assert "HIT_CONFIRMATION_FRAMES = 2" in src
    assert ">= RECORD_DEBOUNCE_SECONDS" in src
