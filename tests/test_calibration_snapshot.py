from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from targetweb import server
from targetweb.targets import load_target


def _build_client(tmp_path: Path) -> TestClient:
    cfg_path = tmp_path / "ha.ini"
    cfg_path.write_text("** Target settings **\n", encoding="utf-8")
    base = Path(server.__file__).resolve().parent
    default_target = (base.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
    tgt = load_target(default_target)
    server.state = server.AppState(latest_dir=tmp_path, target=tgt, caliber_mm=4.5, config_path=cfg_path)
    return TestClient(server.app)


def test_calibration_snapshot_capture_and_live_toggle(tmp_path):
    client = _build_client(tmp_path)
    assert server.state is not None

    # Seed one frame so capture can freeze it.
    server.state.last_raw_frame = np.zeros((120, 160, 3), dtype=np.uint8)

    r = client.post("/api/calibration/capture")
    assert r.status_code == 200
    body = r.json()
    assert body.get("mode") == "snapshot"

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert snap.get("mode") == "snapshot"
    assert snap.get("has_snapshot") is True

    r = client.post("/api/calibration/live")
    assert r.status_code == 200
    assert r.json().get("mode") == "live"

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    assert s.json().get("mode") == "live"

    r = client.post("/api/calibration/clear")
    assert r.status_code == 200
    assert r.json().get("has_snapshot") is False
