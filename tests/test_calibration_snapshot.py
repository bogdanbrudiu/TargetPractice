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
    snap_file = body.get("snapshot_file")
    assert isinstance(snap_file, str) and snap_file
    assert Path(snap_file).exists()

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert snap.get("mode") == "snapshot"
    assert snap.get("has_snapshot") is True
    assert snap.get("snapshot_file") == snap_file
    assert isinstance(snap.get("snapshot_dir"), str)

    r = client.post("/api/calibration/live")
    assert r.status_code == 200
    assert r.json().get("mode") == "live"

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    assert s.json().get("mode") == "live"

    r = client.post("/api/calibration/clear")
    assert r.status_code == 200
    assert r.json().get("has_snapshot") is False
    assert r.json().get("snapshot_count") == 0


def test_calibration_snapshot_multi_capture_and_navigation(tmp_path):
    client = _build_client(tmp_path)
    assert server.state is not None

    server.state.last_raw_frame = np.full((120, 160, 3), 11, dtype=np.uint8)
    r = client.post("/api/calibration/capture")
    assert r.status_code == 200
    assert r.json().get("snapshot_count") == 1
    assert r.json().get("snapshot_index") == 0
    first_file = r.json().get("snapshot_file")
    assert isinstance(first_file, str) and first_file
    assert Path(first_file).exists()

    server.state.last_raw_frame = np.full((120, 160, 3), 22, dtype=np.uint8)
    r = client.post("/api/calibration/capture")
    assert r.status_code == 200
    assert r.json().get("snapshot_count") == 2
    assert r.json().get("snapshot_index") == 1
    second_file = r.json().get("snapshot_file")
    assert isinstance(second_file, str) and second_file
    assert Path(second_file).exists()
    assert second_file != first_file

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert snap.get("mode") == "snapshot"
    assert snap.get("has_snapshot") is True
    assert snap.get("snapshot_count") == 2
    assert snap.get("snapshot_index") == 1
    assert snap.get("can_prev") is True
    assert snap.get("can_next") is False
    assert snap.get("snapshot_file") == second_file

    r = client.post("/api/calibration/snapshot/prev")
    assert r.status_code == 200
    assert r.json().get("snapshot_index") == 0

    # At first snapshot, prev should not move and should return conflict.
    r = client.post("/api/calibration/snapshot/prev")
    assert r.status_code == 409

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert snap.get("snapshot_index") == 0
    assert snap.get("can_prev") is False
    assert snap.get("can_next") is True
    assert snap.get("snapshot_file") == first_file

    r = client.post("/api/calibration/snapshot/next")
    assert r.status_code == 200
    assert r.json().get("snapshot_index") == 1

    # At last snapshot, next should not move and should return conflict.
    r = client.post("/api/calibration/snapshot/next")
    assert r.status_code == 409

    r = client.post("/api/calibration/snapshot/select", json={"index": 0})
    assert r.status_code == 200
    assert r.json().get("snapshot_index") == 0

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert isinstance(snap.get("snapshots"), list)
    assert len(snap.get("snapshots")) == 2
    assert snap.get("snapshot_index") == 0

    # Delete current snapshot only (index 0), one snapshot should remain.
    r = client.post("/api/calibration/clear")
    assert r.status_code == 200
    body = r.json()
    assert body.get("has_snapshot") is True
    assert body.get("snapshot_count") == 1

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert snap.get("snapshot_count") == 1
    assert isinstance(snap.get("snapshots"), list)
    assert len(snap.get("snapshots")) == 1

    # Delete the last remaining snapshot; now snapshot mode falls back to live.
    r = client.post("/api/calibration/clear")
    assert r.status_code == 200
    body = r.json()
    assert body.get("has_snapshot") is False
    assert body.get("snapshot_file") is None
    assert body.get("snapshot_count") == 0
    assert body.get("snapshot_index") == -1


def test_calibration_snapshot_state_loads_existing_files_from_disk(tmp_path):
    client = _build_client(tmp_path)
    assert server.state is not None

    # Create snapshots on disk via API capture.
    server.state.last_raw_frame = np.full((120, 160, 3), 55, dtype=np.uint8)
    r = client.post("/api/calibration/capture")
    assert r.status_code == 200
    server.state.last_raw_frame = np.full((120, 160, 3), 77, dtype=np.uint8)
    r = client.post("/api/calibration/capture")
    assert r.status_code == 200

    # Simulate a fresh in-memory state while files still exist on disk.
    server.state.calib_snapshots = []
    server.state.calib_snapshot_ts = []
    server.state.calib_snapshot_files = []
    server.state.calib_snapshot_index = -1
    server.state.calib_frozen_frame = None
    server.state.calib_frozen_ts = None
    server.state.calib_frozen_file = None
    server.state.calib_use_frozen = False

    s = client.get("/api/calibration/snapshot")
    assert s.status_code == 200
    snap = s.json()
    assert snap.get("snapshot_count") == 2
    assert isinstance(snap.get("snapshots"), list)
    assert len(snap.get("snapshots")) == 2
