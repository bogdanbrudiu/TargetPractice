from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    target_filename: Optional[str] = None
    weapon_filename: Optional[str] = None
    scale: float = 1.0
    position_x: float = 0.0
    position_y: float = 0.0
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    sensitivity: Optional[float] = None
    gain: Optional[float] = None
    red_sat_min: Optional[int] = None
    red_val_min: Optional[int] = None
    red_gate_kernel: Optional[int] = None
    white_sat_max: Optional[int] = None
    white_val_min: Optional[int] = None
    flip_vertical: Optional[int] = None
    flip_horizontal: Optional[int] = None
    distance_simulated: Optional[float] = None
    distance_real: Optional[float] = None
    distance_is_metric: Optional[int] = None
    shooter: Optional[str] = None
    rounds: Optional[int] = None
    timer: Optional[int] = None
    time_limit: Optional[int] = None
    countdown: Optional[int] = None
    hit_limit: Optional[int] = None
    slh: Optional[int] = None
    prepare: Optional[int] = None


@dataclass
class Weapon:
    name: str
    # caliber value as stored in weapon file, in the given unit
    caliber: float
    caliber_mm: float
    unit: str = "mm"


def parse_ini(path: Path) -> AppConfig:
    cfg = AppConfig()
    if not path.exists():
        return cfg
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        if k == "VideoWidth":
            try:
                cfg.video_width = int(float(v))
            except ValueError:
                pass
            continue
        if k == "VideoHeight":
            try:
                cfg.video_height = int(float(v))
            except ValueError:
                pass
            continue
        if k == "Sensitivity":
            try:
                cfg.sensitivity = float(v.replace(",", "."))
            except ValueError:
                pass
            continue
        if k == "Gain":
            try:
                cfg.gain = float(v.replace(",", "."))
            except ValueError:
                pass
            continue
        if k == "RedSatMin":
            try:
                cfg.red_sat_min = int(float(v))
            except ValueError:
                pass
            continue
        if k == "RedValMin":
            try:
                cfg.red_val_min = int(float(v))
            except ValueError:
                pass
            continue
        if k == "RedGateKernel":
            try:
                cfg.red_gate_kernel = int(float(v))
            except ValueError:
                pass
            continue
        if k == "WhiteSatMax":
            try:
                cfg.white_sat_max = int(float(v))
            except ValueError:
                pass
            continue
        if k == "WhiteValMin":
            try:
                cfg.white_val_min = int(float(v))
            except ValueError:
                pass
            continue
        if k == "FlipCamVertical":
            try:
                cfg.flip_vertical = int(float(v))
            except ValueError:
                pass
            continue
        if k == "FlipCamHorizontal":
            try:
                cfg.flip_horizontal = int(float(v))
            except ValueError:
                pass
            continue
        if k == "TargetFilename":
            if v and v.lower() != "null":
                cfg.target_filename = v
        elif k == "WeaponFilename":
            if v:
                cfg.weapon_filename = v
        elif k == "Scale":
            try:
                cfg.scale = float(v)
            except ValueError:
                pass
        elif k == "PositionX":
            try:
                cfg.position_x = float(v)
            except ValueError:
                pass
        elif k == "PositionY":
            try:
                cfg.position_y = float(v)
            except ValueError:
                pass
        elif k == "DistanceSimulated":
            try:
                cfg.distance_simulated = float(v.replace(",", "."))
            except ValueError:
                pass
        elif k == "DistanceReal":
            try:
                cfg.distance_real = float(v.replace(",", "."))
            except ValueError:
                pass
        elif k == "DistanceIsMetric":
            try:
                cfg.distance_is_metric = int(float(v))
            except ValueError:
                pass
        elif k == "Shooter":
            if v:
                cfg.shooter = v
        elif k == "Rounds":
            try:
                cfg.rounds = int(float(v))
            except ValueError:
                pass
        elif k == "Timer":
            try:
                cfg.timer = int(float(v))
            except ValueError:
                pass
        elif k == "TimeLimit":
            try:
                cfg.time_limit = int(float(v))
            except ValueError:
                pass
        elif k == "Countdown":
            try:
                cfg.countdown = int(float(v))
            except ValueError:
                pass
        elif k == "HitLimit":
            try:
                cfg.hit_limit = int(float(v))
            except ValueError:
                pass
        elif k == "SLH":
            try:
                cfg.slh = int(float(v))
            except ValueError:
                pass
        elif k == "Prepare":
            try:
                cfg.prepare = int(float(v))
            except ValueError:
                pass
    return cfg


def parse_weapon(path: Path) -> Weapon:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    name = lines[0] if lines else path.stem
    cal_value = 4.5
    unit = "mm"
    if len(lines) >= 3:
        try:
            cal_value = float(lines[1].replace(",", "."))
        except ValueError:
            cal_value = 4.5
        unit = lines[2].lower()
    if unit in ("inch", "inches"):
        cal_value_mm = cal_value * 25.4
    else:
        cal_value_mm = cal_value
    unit_norm = ("inch" if unit in ("inch", "inches") else "mm")
    return Weapon(name=name, caliber=cal_value, caliber_mm=cal_value_mm, unit=unit_norm)


def write_weapon(path: Path, name: str, caliber: float, unit: str) -> None:
    """Write weapon file preserving trailing lines after third line."""
    unit_norm = "inch" if unit.lower() in ("inch", "inches") else "mm"
    lines = path.read_text(encoding="utf-8").splitlines()
    rest = lines[3:] if len(lines) > 3 else []
    out = [name, f"{caliber}", unit_norm]
    out.extend(rest)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
