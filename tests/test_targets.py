from pathlib import Path
from targetweb.targets import load_target, CircleTarget, MonoscopeTarget


BASE = Path(__file__).resolve().parent.parent


def test_parse_air_pistol():
    p = BASE / "data" / "targets" / "10m_air_pistol.tgt"
    tgt = load_target(p)
    assert isinstance(tgt, CircleTarget)
    assert tgt.info.display_name.lower().startswith("10m air pistol")
    assert len(tgt.rings) == 10


def test_parse_monoscope():
    p = BASE / "data" / "targets" / "monoscope_640.tgt"
    tgt = load_target(p)
    assert isinstance(tgt, MonoscopeTarget)
    assert tgt.info.type == 4
