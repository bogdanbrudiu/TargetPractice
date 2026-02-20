from pathlib import Path

import numpy as np

from targetweb import server
from targetweb.targets import load_target


def test_regenerate_overlay_respects_slh_last_n(tmp_path, monkeypatch):
    # Arrange: create a fresh state with a temp config and two shots
    cfg_path = Path(tmp_path) / "ha.ini"
    # With SLH=1, overlay should draw only the last shot.
    cfg_path.write_text("** Target settings **\nSLH = 1\n", encoding="utf-8")

    base = Path(server.__file__).resolve().parent
    default_target = (base.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
    tgt = load_target(default_target)

    st = server.AppState(latest_dir=Path(tmp_path), target=tgt, caliber_mm=4.5, config_path=cfg_path)
    st.last_raw_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with st.session_lock:
        st.session.add(10, 20, 5.0)
        st.session.add(30, 40, 7.5)

    server.state = st

    captured = {}

    def fake_draw_overlay(frame_bgr, hints, hit_xy, score, caliber_mm=None, pixels_per_mm=None, shots=None):
        captured["shots"] = shots
        return frame_bgr

    def fake_save_rendered_image(img):
        # avoid filesystem writes
        return None

    monkeypatch.setattr(server, "draw_overlay", fake_draw_overlay)
    monkeypatch.setattr(server, "_save_rendered_image", fake_save_rendered_image)

    # Act
    server._regenerate_from_last_raw()

    # Assert: only last shot is drawn
    shots = captured.get("shots")
    assert shots is not None
    assert len(shots) == 1
    assert shots[0]["i"] == 2


def test_regenerate_overlay_draws_all_when_slh_zero(tmp_path, monkeypatch):
    cfg_path = Path(tmp_path) / "ha.ini"
    cfg_path.write_text("** Target settings **\nSLH = 0\n", encoding="utf-8")

    base = Path(server.__file__).resolve().parent
    default_target = (base.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
    tgt = load_target(default_target)

    st = server.AppState(latest_dir=Path(tmp_path), target=tgt, caliber_mm=4.5, config_path=cfg_path)
    st.last_raw_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with st.session_lock:
        st.session.add(10, 20, 5.0)
        st.session.add(30, 40, 7.5)

    server.state = st

    captured = {}

    def fake_draw_overlay(frame_bgr, hints, hit_xy, score, caliber_mm=None, pixels_per_mm=None, shots=None):
        captured["shots"] = shots
        return frame_bgr

    def fake_save_rendered_image(img):
        return None

    monkeypatch.setattr(server, "draw_overlay", fake_draw_overlay)
    monkeypatch.setattr(server, "_save_rendered_image", fake_save_rendered_image)

    server._regenerate_from_last_raw()

    shots = captured.get("shots")
    assert shots is not None
    assert len(shots) == 2
    assert shots[0]["i"] == 1
    assert shots[1]["i"] == 2
