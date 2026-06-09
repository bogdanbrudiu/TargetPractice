import csv
from pathlib import Path

import numpy as np

from targetweb import server
from targetweb.detector import Hit
from targetweb.targets import load_target


def test_record_debug_hit_writes_image_and_csv(tmp_path):
    cfg_path = tmp_path / "ha.ini"
    cfg_path.write_text("Sensitivity = 1\nGain = 10\n", encoding="utf-8")

    base = Path(server.__file__).resolve().parent
    default_target = (base.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
    tgt = load_target(default_target)
    app_state = server.AppState(latest_dir=tmp_path, target=tgt, caliber_mm=4.5, config_path=cfg_path)
    app_state.shooting_status = "running"
    app_state.active_weapon_filename = "TAU7_4_5mm.gun"
    app_state.detector_threshold = 252

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    hit = Hit(x=31.6, y=22.4, strength=2.0)

    server._record_debug_hit(app_state, frame, hit, 9.5, "auto", ts=1717855200.123)

    files = sorted(app_state.hit_debug_dir.glob("*.jpg"))
    assert len(files) == 1
    assert "_auto_" in files[0].name
    assert "_x32_y22_" in files[0].name

    with app_state.hit_debug_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["image_file"] == files[0].name
    assert rows[0]["source"] == "auto"
    assert rows[0]["x"] == "31.60"
    assert rows[0]["y"] == "22.40"
    assert rows[0]["score"] == "9.5"
    assert rows[0]["label"] == ""
    assert rows[0]["notes"] == ""