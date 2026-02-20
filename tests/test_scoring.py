from pathlib import Path
from targetweb.targets import load_target, CircleTarget


BASE = Path(__file__).resolve().parent.parent


def test_scoring_center_is_max():
    p = BASE / "data" / "targets" / "10m_air_rifle.tgt"
    tgt = load_target(p)
    # simulate 640x480 frame
    tgt.auto_scale((640, 480))
    tgt.set_position(320, 240)
    # hit center, small caliber
    s = tgt.get_points(320, 240, caliber_mm=4.5)
    assert s == tgt.rings[-1].points
