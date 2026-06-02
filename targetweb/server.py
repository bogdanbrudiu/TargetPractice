from __future__ import annotations
import argparse
import os
import socket
import time
import math
from pathlib import Path
from typing import Optional, List, Tuple
from threading import Lock

import uvicorn
from fastapi import FastAPI, Request
from fastapi import Body
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
import shutil

from .targets import load_target, ITarget
from .detector import USBCameraSource, BrightSpotDetector, Hit, seeded_component_mask
from .overlay import draw_overlay
from .config import parse_ini, parse_weapon, write_weapon, AppConfig
from .session import Session
from .hid_input import HIDConfig, HIDListener


# Minimum time between auto-recorded hits (to avoid multi-frame glare flooding)
RECORD_DEBOUNCE_SECONDS = 0.10
# Ignore detections briefly right after entering running state.
SHOOTING_START_GRACE_SECONDS = 0.50
# Require at least N close detections across consecutive frames before recording.
HIT_CONFIRMATION_FRAMES = 1
# Max pixel drift between consecutive hits to treat them as the same candidate.
HIT_CONFIRMATION_DISTANCE_PX = 20.0
# Re-arm auto-record only after this many consecutive no-hit frames.
HIT_RELEASE_FRAMES = 2


class AppState:
    def __init__(self, latest_dir: Path, target: ITarget, caliber_mm: float, config_path: Optional[Path] = None) -> None:
        self.latest_dir = latest_dir
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = latest_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.latest_jpg = self.frames_dir / "latest.jpg"
        self.seq = 0
        self.target = target
        self.caliber_mm = caliber_mm
        self.last_hit: Optional[Hit] = None
        self.config_path: Optional[Path] = config_path
        self.last_frame_path: Optional[Path] = None
        self.last_frame_name: Optional[str] = None
        self.last_raw_frame: Optional[np.ndarray] = None
        self.calib_snapshots: List[np.ndarray] = []
        self.calib_snapshot_ts: List[float] = []
        self.calib_snapshot_files: List[Optional[str]] = []
        self.calib_snapshot_index: int = -1
        self.calib_frozen_frame: Optional[np.ndarray] = None
        self.calib_frozen_ts: Optional[float] = None
        self.calib_frozen_file: Optional[str] = None
        self.calib_use_frozen: bool = False
        self.calib_debug_dir = latest_dir / "calib_debug"
        self.calib_debug_dir.mkdir(parents=True, exist_ok=True)
        self.calib_snapshots_dir = self.calib_debug_dir / "snapshots"
        self.calib_snapshots_dir.mkdir(parents=True, exist_ok=True)
        # detector tuning (for calibration/debug UI)
        self.detector_threshold: int = 230
        self.detector_min_area: int = 5
        self.detector_red_sat_min: int = 70
        self.detector_red_val_min: int = 70
        self.detector_red_gate_kernel: int = 7
        self.detector_white_sat_max: int = 16
        self.detector_white_val_min: int = 240
        self.write_lock: Lock = Lock()
        self.session_lock: Lock = Lock()
        # baseline (set on first frame after auto-scale & centering)
        self.initial_p2mm: Optional[float] = None
        self.base_center: Optional[Tuple[float, float]] = None
        self.frame_size: Optional[Tuple[int, int]] = None  # (w,h)
        # shooting/session state (HomeLESS-like)
        self.shooting_status: str = "idle"  # idle|prepare|running|finished
        self.shooting_prepare_end_ts: Optional[float] = None
        self.shooting_running_start_ts: Optional[float] = None
        self.shooting_start_shot_index: int = 0
        self.shooting_time_limit_enabled: bool = False
        self.shooting_time_limit_seconds: int = 0
        self.shooting_time_countdown_enabled: bool = False
        self.shooting_hit_limit_enabled: bool = False
        self.shooting_hit_limit_rounds: int = 0
        self.last_recorded_ts: float = 0.0
        # One-shot latch to avoid repeated records from persistent hotspots.
        self.hit_record_armed: bool = True
        self.hit_nohit_frames: int = 0
        # Ensure a placeholder image exists so /frame never returns JSON
        try:
            if not self.latest_jpg.exists():
                import numpy as _np
                import cv2 as _cv2
                _img = _np.zeros((480, 640, 3), dtype=_np.uint8)
                _cv2.imwrite(str(self.latest_jpg), _img)
        except Exception:
            pass
        # session CSV mirrors HomeLESS format, stored under latest_dir/sessions
        self.sessions_dir = latest_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        cx, cy = target.info.position_px
        self.session = Session(out_dir=self.sessions_dir, target_filename=target.info.filename, cx=cx, cy=cy, scale=target.info.pixel_to_mm_scale)
        # active selection
        self.active_target_filename: str = target.info.filename
        self.active_weapon_filename: Optional[str] = None


app = FastAPI()
BASE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE.parent / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE.parent / "static")), name="static")
state: Optional[AppState] = None

# Default latest directory for placeholder serving before background loop starts
DEFAULT_LATEST_DIR = (BASE.parent / "run" / "frames")
DEFAULT_LATEST_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LATEST = DEFAULT_LATEST_DIR / "latest.jpg"

def _ensure_placeholder(path: Path) -> None:
    try:
        if not path.exists():
            import numpy as _np
            import cv2 as _cv2
            _img = _np.zeros((480, 640, 3), dtype=_np.uint8)
            _cv2.putText(_img, "Starting...", (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, _cv2.LINE_AA)
            _cv2.imwrite(str(path), _img)
    except Exception:
        pass

_ensure_placeholder(DEFAULT_LATEST)


# Helper: list files in data dirs (targets/weapons)
def _list_data_files(subdir: str, pattern: str = "*") -> List[str]:
    base = BASE.parent / "data" / subdir
    if not base.exists():
        return []
    return [p.name for p in sorted(base.glob(pattern))]


def _save_scale_position_to_config(state: AppState) -> None:
    if state.config_path is None or state.initial_p2mm is None or state.base_center is None or state.frame_size is None:
        return
    try:
        cfg_path = state.config_path
        txt = cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        in_target = False
        saw_scale = False
        saw_px = False
        saw_py = False
        # compute values
        current_p2mm = state.target.info.pixel_to_mm_scale
        initial = max(state.initial_p2mm, 1e-6)
        cfg_scale = current_p2mm / initial
        cx, cy = state.target.info.position_px
        bw, bh = state.base_center
        posx = int(round(cx - bw))
        posy = int(round(cy - bh))
        for line in txt:
            line_stripped = line.strip()
            if line_stripped.startswith("**") and "Target settings" in line_stripped:
                in_target = True
                out.append(line)
                continue
            if in_target:
                if line_stripped.startswith("**") and "Target settings" not in line_stripped and line_stripped.startswith("**"):
                    # leaving section
                    if not saw_scale:
                        out.append(f"Scale = {cfg_scale:.4f}")
                    if not saw_px:
                        out.append(f"PositionX = {posx}")
                    if not saw_py:
                        out.append(f"PositionY = {posy}")
                    in_target = False
                    out.append(line)
                    continue
                if line_stripped.startswith("Scale"):
                    out.append(f"Scale = {cfg_scale:.4f}")
                    saw_scale = True
                    continue
                if line_stripped.startswith("PositionX"):
                    out.append(f"PositionX = {posx}")
                    saw_px = True
                    continue
                if line_stripped.startswith("PositionY"):
                    out.append(f"PositionY = {posy}")
                    saw_py = True
                    continue
            out.append(line)
        # if we never encountered the section end (e.g., section at EOF)
        if in_target:
            if not saw_scale:
                out.append(f"Scale = {cfg_scale:.4f}")
            if not saw_px:
                out.append(f"PositionX = {posx}")
            if not saw_py:
                out.append(f"PositionY = {posy}")
        cfg_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    except Exception as ex:
        print(f"Config save failed: {ex}")


def _apply_scale_position(state: Optional[AppState], scale: Optional[float], posx: Optional[float], posy: Optional[float]) -> None:
    """Apply config Scale (relative to baseline) and Position offsets to the live target."""
    if state is None:
        return
    try:
        if scale is not None and state.initial_p2mm is not None:
            base = max(state.initial_p2mm, 1e-6)
            state.target.set_scale(base * scale)
        if posx is not None and posy is not None and state.base_center is not None:
            if abs(posx) < 4000 and abs(posy) < 4000:
                bx, by = state.base_center
                state.target.set_position(bx + posx, by + posy)
    except Exception:
        pass


def _save_scale_only_to_config(state: AppState) -> None:
    if state.config_path is None or state.initial_p2mm is None:
        return
    try:
        current_p2mm = state.target.info.pixel_to_mm_scale
        initial = max(state.initial_p2mm, 1e-6)
        cfg_scale = current_p2mm / initial
        _save_settings_to_ini(state.config_path, {"Scale": float(f"{cfg_scale:.4f}")})
    except Exception as ex:
        print(f"Config save failed: {ex}")


def _save_rendered_image(img: np.ndarray) -> None:
    global state
    if state is None:
        return
    frames_dir = state.latest_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with state.write_lock:
        state.seq += 1
        unique = frames_dir / f"latest_{state.seq}.jpg"
        try:
            cv2.imwrite(str(unique), img)
            state.last_frame_path = unique
            state.last_frame_name = unique.name
            # also update canonical latest.jpg for clients expecting that name
            tmp_latest = frames_dir / f".latest_tmp_{state.seq}.jpg"
            try:
                cv2.imwrite(str(tmp_latest), img)
                os.replace(str(tmp_latest), str(state.latest_jpg))
            except PermissionError:
                # can't replace, best effort cleanup
                try:
                    tmp_latest.unlink(missing_ok=True)
                except Exception:
                    pass
            # prune all previous frames, keep only the newest one
            try:
                for pattern in ("latest_*.jpg", "frame_*.jpg"):
                    for p in frames_dir.glob(pattern):
                        if p.name != state.last_frame_name:
                            try:
                                p.unlink()
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            # best effort cleanup on failure
            try:
                if 'unique' in locals():
                    unique.unlink(missing_ok=True)
            except Exception:
                pass


def _regenerate_from_last_raw() -> None:
    global state
    if state is None:
        return
    if state.last_raw_frame is None:
        return

    # Draw recorded shots on the overlay.
    # HomeLESS-like: SLH controls how many recent shots are shown.
    shots_to_draw: list[dict] = []
    try:
        with state.session_lock:
            raw = list(getattr(state.session, "shots", []) or [])

        n = 0
        try:
            if state.config_path and state.config_path.exists():
                cfg = parse_ini(state.config_path)
                n = int(cfg.slh or 0)
        except Exception:
            n = 0

        if n > 0:
            start = max(0, len(raw) - n)
            shots_to_draw = [{"x": s.x, "y": s.y, "i": (i + 1)} for i, s in enumerate(raw[start:], start=start)]
        else:
            shots_to_draw = [{"x": s.x, "y": s.y, "i": (i + 1)} for i, s in enumerate(raw)]
    except Exception:
        shots_to_draw = []

    img = draw_overlay(
        state.last_raw_frame,
        state.target.overlay_hint(),
        None,
        None,
        caliber_mm=getattr(state, "caliber_mm", None),
        pixels_per_mm=getattr(state.target.info, "pixel_to_mm_scale", None),
        shots=shots_to_draw,
    )
    _save_rendered_image(img)


def _jpeg_response(img: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return Response(content=b"", media_type="image/jpeg", status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def _shooting_snapshot(now_ts: float) -> dict:
    """Return current shooting state for UI/polling."""
    global state
    if state is None:
        return {
            "shooting_status": "idle",
            "shooting_prepare_remaining": 0,
            "shooting_elapsed": 0.0,
            "shooting_time_remaining": None,
            "shooting_hits": 0,
            "shooting_hits_remaining": None,
            "shooting_can_record": False,
        }

    with state.session_lock:
        total_shots = len(getattr(state.session, "shots", []))

    status = getattr(state, "shooting_status", "idle")
    prepare_end = getattr(state, "shooting_prepare_end_ts", None)
    running_start = getattr(state, "shooting_running_start_ts", None)
    start_idx = int(getattr(state, "shooting_start_shot_index", 0))

    shots_since_start = max(0, total_shots - start_idx)

    prepare_remaining = 0
    if status == "prepare" and prepare_end is not None:
        prepare_remaining = max(0, int(math.ceil(prepare_end - now_ts)))

    elapsed = 0.0
    if running_start is not None and status in ("running", "finished"):
        elapsed = max(0.0, now_ts - running_start)

    time_remaining = None
    if getattr(state, "shooting_time_limit_enabled", False):
        limit = int(getattr(state, "shooting_time_limit_seconds", 0))
        if limit > 0 and running_start is not None:
            time_remaining = max(0, int(round(limit - elapsed)))

    hits_remaining = None
    if getattr(state, "shooting_hit_limit_enabled", False):
        rounds = int(getattr(state, "shooting_hit_limit_rounds", 0))
        if rounds > 0:
            hits_remaining = max(0, rounds - shots_since_start)

    can_record = status == "running"
    return {
        "shooting_status": status,
        "shooting_prepare_remaining": prepare_remaining,
        "shooting_elapsed": elapsed,
        "shooting_time_remaining": time_remaining,
        "shooting_time_countdown": bool(getattr(state, "shooting_time_countdown_enabled", False)),
        "shooting_hits": shots_since_start,
        "shooting_hits_remaining": hits_remaining,
        "shooting_can_record": can_record,
    }


def _shooting_tick(now_ts: float) -> None:
    """Advance shooting state machine and finish when limits reached.

    Called from both the capture loop and /api/state so the state progresses
    even when no new frames arrive (e.g. static source exhausted).
    """
    global state
    if state is None:
        return
    with state.session_lock:
        status = getattr(state, "shooting_status", "idle")
        if status == "prepare":
            end_ts = getattr(state, "shooting_prepare_end_ts", None)
            if end_ts is not None and now_ts >= end_ts:
                state.shooting_status = "running"
                state.shooting_running_start_ts = now_ts
                state.last_recorded_ts = now_ts
                state.last_hit = None
                state.hit_record_armed = True
                state.hit_nohit_frames = 0

        if getattr(state, "shooting_status", "idle") != "running":
            return

        start_ts = getattr(state, "shooting_running_start_ts", None)
        if start_ts is None:
            state.shooting_running_start_ts = now_ts
            start_ts = now_ts

        # stop by time limit
        if getattr(state, "shooting_time_limit_enabled", False):
            lim = int(getattr(state, "shooting_time_limit_seconds", 0))
            if lim > 0 and (now_ts - start_ts) >= lim:
                state.shooting_status = "finished"
                return

        # stop by hit limit
        if getattr(state, "shooting_hit_limit_enabled", False):
            rounds = int(getattr(state, "shooting_hit_limit_rounds", 0))
            if rounds > 0:
                total_shots = len(getattr(state.session, "shots", []))
                start_idx = int(getattr(state, "shooting_start_shot_index", 0))
                shots_since = max(0, total_shots - start_idx)
                if shots_since >= rounds:
                    state.shooting_status = "finished"


@app.get("/")
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/frame")
async def frame():
    # Prefer serving the latest unique frame path to avoid file lock issues
    if state is not None and state.last_frame_path is not None and state.last_frame_path.exists():
        return FileResponse(path=str(state.last_frame_path), media_type="image/jpeg")
    if state is not None and state.latest_jpg.exists():
        return FileResponse(path=str(state.latest_jpg), media_type="image/jpeg")
    # serve placeholder until state initializes
    _ensure_placeholder(DEFAULT_LATEST)
    return FileResponse(path=str(DEFAULT_LATEST), media_type="image/jpeg")


@app.get("/frames/{name}")
async def frames(name: str):
    if state is not None:
        path = state.frames_dir / name
        if path.exists():
            return FileResponse(path=str(path), media_type="image/jpeg")
    # allow placeholder "latest.jpg" before init
    if name == "latest.jpg" and DEFAULT_LATEST.exists():
        return FileResponse(path=str(DEFAULT_LATEST), media_type="image/jpeg")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/state")
async def api_state():
    # advance shooting state even if capture loop isn't producing frames
    try:
        _shooting_tick(time.time())
    except Exception:
        pass
    if state is None:
        # try to read active selection from config file so UI can show it before full startup
        cfg_path = BASE.parent / "config" / "ha.ini"
        active_t = None
        active_w = None
        extra = {}
        try:
            if cfg_path.exists():
                cfg = _read_target_settings(cfg_path)
                active_t = cfg.get("TargetFilename")
                active_w = cfg.get("WeaponFilename")
                # include target scale/position and weapon fields from config when uninitialized
                extra = {
                    "VideoWidth": cfg.get("VideoWidth"),
                    "VideoHeight": cfg.get("VideoHeight"),
                    "Sensitivity": cfg.get("Sensitivity"),
                    "Gain": cfg.get("Gain"),
                    "RedSatMin": cfg.get("RedSatMin"),
                    "RedValMin": cfg.get("RedValMin"),
                    "RedGateKernel": cfg.get("RedGateKernel"),
                    "WhiteSatMax": cfg.get("WhiteSatMax"),
                    "WhiteValMin": cfg.get("WhiteValMin"),
                    "Scale": cfg.get("Scale"),
                    "PositionX": cfg.get("PositionX"),
                    "PositionY": cfg.get("PositionY"),
                    "Shooter": cfg.get("Shooter"),
                    "Rounds": cfg.get("Rounds"),
                    "Timer": cfg.get("Timer"),
                    "TimeLimit": cfg.get("TimeLimit"),
                    "Countdown": cfg.get("Countdown"),
                    "HitLimit": cfg.get("HitLimit"),
                    "SLH": cfg.get("SLH"),
                    "Prepare": cfg.get("Prepare"),
                }
        except Exception:
            pass
        base = {"total": 0.0, "shots": [], "frame_name": "latest.jpg", "active_target": active_t, "active_weapon": active_w}
        base.update(extra)
        base.update(_shooting_snapshot(time.time()))
        return JSONResponse(base)
    session = state.session
    # compute config-like Scale/PositionX/PositionY from current state (fallback to config if baseline missing)
    cfg_scale = None
    posx = None
    posy = None
    try:
        if state.initial_p2mm is not None:
            current_p2mm = state.target.info.pixel_to_mm_scale
            initial = max(state.initial_p2mm, 1e-6)
            cfg_scale = current_p2mm / initial
        if state.base_center is not None:
            cx, cy = state.target.info.position_px
            bw, bh = state.base_center
            posx = int(round(cx - bw))
            posy = int(round(cy - bh))
    except Exception:
        pass
    # read config for weapon fields and fallback values
    cfg_fields = {}
    try:
        if state.config_path and state.config_path.exists():
            cfg = _read_target_settings(state.config_path)
            cfg_fields = {
                "VideoWidth": cfg.get("VideoWidth"),
                "VideoHeight": cfg.get("VideoHeight"),
                "Sensitivity": cfg.get("Sensitivity"),
                "Gain": cfg.get("Gain"),
                "RedSatMin": cfg.get("RedSatMin"),
                "RedValMin": cfg.get("RedValMin"),
                "RedGateKernel": cfg.get("RedGateKernel"),
                "WhiteSatMax": cfg.get("WhiteSatMax"),
                "WhiteValMin": cfg.get("WhiteValMin"),
                "Shooter": cfg.get("Shooter"),
                "Rounds": cfg.get("Rounds"),
                "Timer": cfg.get("Timer"),
                "TimeLimit": cfg.get("TimeLimit"),
                "Countdown": cfg.get("Countdown"),
                "HitLimit": cfg.get("HitLimit"),
                "SLH": cfg.get("SLH"),
                "Prepare": cfg.get("Prepare"),
            }
            # fill missing scale/position from config
            if cfg_scale is None:
                cfg_scale = cfg.get("Scale")
            if posx is None:
                posx = cfg.get("PositionX")
            if posy is None:
                posy = cfg.get("PositionY")
    except Exception:
        pass
    with state.session_lock:
        shots_list_raw = getattr(session, 'shots', [])
        total_val = getattr(session, 'total', 0.0)

    # Ensure shots are JSON-serializable (Session stores Shot dataclasses).
    shots_list = []
    try:
        for s in (shots_list_raw or []):
            if isinstance(s, dict):
                shots_list.append(s)
            else:
                shots_list.append({
                    "x": float(getattr(s, "x")),
                    "y": float(getattr(s, "y")),
                    "score": float(getattr(s, "score")),
                    **({"time": getattr(s, "time")} if getattr(s, "time", None) is not None else {}),
                })
    except Exception:
        shots_list = []

    out = {
        "total": total_val,
        "shots": shots_list,
        "frame_name": state.last_frame_name or (state.latest_jpg.name if state.latest_jpg else "latest.jpg"),
        "active_target": getattr(state, 'active_target_filename', None),
        "active_weapon": getattr(state, 'active_weapon_filename', None),
        "Scale": cfg_scale,
        "PositionX": posx,
        "PositionY": posy,
    }
    out.update(cfg_fields)
    out.update(_shooting_snapshot(time.time()))
    return JSONResponse(out)


@app.post("/api/shooting/start")
async def api_shooting_start():
    """Start a shooting session (HomeLESS-like): prepare countdown then enable hit recording."""
    global state
    if state is None or state.config_path is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)

    # snapshot current config and freeze for this shooting run
    cfg = parse_ini(state.config_path) if state.config_path.exists() else AppConfig()
    # Prepare=0 should start immediately.
    try:
        prepare_seconds = int(cfg.prepare if cfg.prepare is not None else 0)
    except Exception:
        prepare_seconds = 0
    if prepare_seconds < 0:
        prepare_seconds = 0

    time_limit_enabled = bool(int(cfg.time_limit or 0))
    time_limit_seconds = int(cfg.timer or 0)
    if time_limit_enabled and time_limit_seconds <= 0:
        time_limit_seconds = 1

    hit_limit_rounds = int(cfg.rounds or 0)
    hit_limit_enabled_cfg = bool(int(cfg.hit_limit or 0))
    # Rounds>0 should enforce an auto-stop even if HitLimit checkbox is off.
    hit_limit_enabled = hit_limit_enabled_cfg or (hit_limit_rounds > 0)
    if hit_limit_enabled and hit_limit_rounds <= 0:
        hit_limit_rounds = 1

    now_ts = time.time()
    with state.session_lock:
        state.shooting_start_shot_index = len(getattr(state.session, "shots", []))
        state.shooting_time_limit_enabled = time_limit_enabled
        state.shooting_time_limit_seconds = time_limit_seconds
        state.shooting_time_countdown_enabled = bool(int(cfg.countdown or 0))
        state.shooting_hit_limit_enabled = hit_limit_enabled
        state.shooting_hit_limit_rounds = hit_limit_rounds
        state.shooting_running_start_ts = None
        state.shooting_prepare_end_ts = now_ts + prepare_seconds
        state.shooting_status = "prepare" if prepare_seconds > 0 else "running"
        state.hit_record_armed = True
        state.hit_nohit_frames = 0
        if state.shooting_status == "running":
            state.shooting_running_start_ts = now_ts
            state.last_recorded_ts = now_ts
        else:
            state.last_recorded_ts = 0.0
        state.last_hit = None

    try:
        _shooting_tick(now_ts)
    except Exception:
        pass
    return JSONResponse({"ok": True, **_shooting_snapshot(now_ts)})


@app.post("/api/shooting/stop")
async def api_shooting_stop():
    global state
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    with state.session_lock:
        state.shooting_status = "idle"
        state.shooting_prepare_end_ts = None
        state.shooting_running_start_ts = None
        state.hit_record_armed = True
        state.hit_nohit_frames = 0
    return JSONResponse({"ok": True, **_shooting_snapshot(time.time())})


@app.post("/api/shooting/reset")
async def api_shooting_reset():
    """Reset results (new CSV + clear shots)."""
    global state
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    with state.session_lock:
        cx, cy = state.target.info.position_px
        state.session = Session(
            out_dir=state.sessions_dir,
            target_filename=state.target.info.filename,
            cx=cx,
            cy=cy,
            scale=state.target.info.pixel_to_mm_scale,
        )
        state.last_hit = None
        state.shooting_status = "idle"
        state.shooting_prepare_end_ts = None
        state.shooting_running_start_ts = None
        state.shooting_start_shot_index = 0
        state.hit_record_armed = True
        state.hit_nohit_frames = 0
        state.last_recorded_ts = 0.0
    return JSONResponse({"ok": True, **_shooting_snapshot(time.time())})


@app.get("/api/target")
async def api_target():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    info = state.target.info
    return JSONResponse({
        "filename": info.filename,
        "display_name": info.display_name,
        "type": info.type,
        "is_metric": info.is_metric,
        "position_px": info.position_px,
        "pixel_to_mm_scale": info.pixel_to_mm_scale,
    })


@app.get("/api/calibration/frame")
async def api_calibration_frame(view: str = "thresh"):
    """Calibration/debug frame for tuning sensitivity/gain.

    view: thresh|gray|raw|red|hybrid
    """
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    frame = None
    try:
        if getattr(state, "calib_use_frozen", False) and state.calib_frozen_frame is not None:
            frame = state.calib_frozen_frame.copy()
        elif state.last_raw_frame is not None:
            frame = state.last_raw_frame.copy()
    except Exception:
        frame = state.last_raw_frame
    if frame is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)

    # Apply the same gain multiplier as run_loop (best-effort)
    gain_alpha = None
    try:
        if state.config_path and state.config_path.exists():
            cfg = parse_ini(state.config_path)
            if cfg.gain is not None:
                gain_alpha = max(0.1, float(cfg.gain) / 10.0)
    except Exception:
        gain_alpha = None
    if gain_alpha is not None:
        try:
            frame = cv2.convertScaleAbs(frame, alpha=gain_alpha, beta=0)
        except Exception:
            pass

    if view == "raw":
        out = frame
        cv2.putText(out, "RAW", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return _jpeg_response(out)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if view == "gray":
        out = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        cv2.putText(out, "GRAY/BLUR", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return _jpeg_response(out)

    thr = int(getattr(state, "detector_threshold", 230))
    min_area = int(getattr(state, "detector_min_area", 5))
    red_sat_min = int(getattr(state, "detector_red_sat_min", 70))
    red_val_min = int(getattr(state, "detector_red_val_min", 70))
    red_gate_kernel = int(getattr(state, "detector_red_gate_kernel", 7))
    white_sat_max = int(getattr(state, "detector_white_sat_max", 16))
    white_val_min = int(getattr(state, "detector_white_val_min", 240))
    if red_gate_kernel < 1:
        red_gate_kernel = 1
    if (red_gate_kernel % 2) == 0:
        red_gate_kernel += 1
    img_area = float(frame.shape[0] * frame.shape[1])
    max_area = max(float(min_area), img_area * 0.02)

    def _draw_calibration_detection_marker(out_img: np.ndarray, center: Optional[Tuple[int, int]], detected: bool) -> None:
        h, w = out_img.shape[:2]
        badge_w = min(w - 12, 320)
        badge_y1 = max(0, h - 44)
        cv2.rectangle(out_img, (6, badge_y1), (6 + badge_w, h - 8), (0, 0, 0), -1)
        if detected and center is not None:
            cx, cy = center
            r = max(10, min(h, w) // 24)
            cv2.circle(out_img, (cx, cy), r + 7, (0, 0, 0), 3)
            cv2.circle(out_img, (cx, cy), r + 4, (0, 255, 255), 3)
            cv2.circle(out_img, (cx, cy), r, (0, 0, 255), 2)
            cv2.drawMarker(out_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, r * 2, 2)
            cv2.putText(out_img, "SPOT DETECTED", (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(out_img, "NO SPOT DETECTED", (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2, cv2.LINE_AA)

    if view == "red":
        # Mirror the detector's red masking (HSV) so tuning is easier.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = (0, red_sat_min, red_val_min)
        upper1 = (10, 255, 255)
        lower2 = (170, red_sat_min, red_val_min)
        upper2 = (179, 255, 255)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Mirror detector contour filtering so you can see what would be accepted.
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        total_cnts = len(cnts)
        kept = 0
        best = None
        best_center = None
        best_area = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < float(min_area) or area > float(max_area):
                continue
            kept += 1
            if area > best_area:
                best = c
                best_area = area
            cv2.drawContours(out, [c], -1, (0, 255, 0), 1)

        if best is not None:
            M = cv2.moments(best)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_center = (cx, cy)

        _draw_calibration_detection_marker(out, best_center, best_center is not None)

        cv2.putText(
            out,
            f"RED MASK  SAT>={red_sat_min} VAL>={red_val_min}  MIN_AREA={min_area}  CNT={total_cnts}  KEPT={kept}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return _jpeg_response(out)

    if view == "hybrid":
        # Mirror detector's hybrid path: threshold AND dilated red gate.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = (0, red_sat_min, red_val_min)
        upper1 = (10, 255, 255)
        lower2 = (170, red_sat_min, red_val_min)
        upper2 = (179, 255, 255)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        red_mask = cv2.bitwise_or(m1, m2)
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        red_gate = cv2.dilate(red_mask, np.ones((red_gate_kernel, red_gate_kernel), np.uint8), iterations=1)
        white_mask = cv2.inRange(
            hsv,
            (0, 0, white_val_min),
            (179, white_sat_max, 255),
        )
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
        _, white_local = cv2.threshold(top_hat, 16, 255, cv2.THRESH_BINARY)
        white_mask = cv2.bitwise_and(white_mask, white_local)
        color_gate = cv2.bitwise_or(red_gate, white_mask)

        _, th = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
        seed = cv2.bitwise_and(th, color_gate)
        hybrid = seed
        if cv2.countNonZero(hybrid) <= 0:
            red_seed = cv2.bitwise_and(th, red_gate)
            if cv2.countNonZero(red_seed) > 0:
                hybrid = seeded_component_mask(th, red_seed, max_growth=max(2, red_gate_kernel // 2))
        hybrid = cv2.dilate(hybrid, kernel, iterations=1)

        cnts, _ = cv2.findContours(hybrid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = cv2.cvtColor(hybrid, cv2.COLOR_GRAY2BGR)
        total_cnts = len(cnts)
        kept = 0
        best_center = None
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < float(min_area) or area > float(max_area):
                continue
            kept += 1
            cv2.drawContours(out, [c], -1, (0, 255, 0), 1)

        # Use the same runtime detector logic for the marker center so preview and runtime match 1:1.
        det_dbg = BrightSpotDetector(
            threshold=thr,
            min_area=min_area,
            red_sat_min=red_sat_min,
            red_val_min=red_val_min,
            red_gate_kernel=red_gate_kernel,
            white_sat_max=white_sat_max,
            white_val_min=white_val_min,
        )
        hit_dbg = det_dbg.detect(frame)
        if hit_dbg is not None:
            best_center = (int(round(hit_dbg.x)), int(round(hit_dbg.y)))

        _draw_calibration_detection_marker(out, best_center, best_center is not None)

        cv2.putText(
            out,
            f"HYBRID=T&(RED|WHITE)  RED(k={red_gate_kernel},S>={red_sat_min},V>={red_val_min}) WHITE(S<={white_sat_max},V>={white_val_min}) THRESH={thr} MIN_AREA={min_area} CNT={total_cnts} KEPT={kept}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return _jpeg_response(out)

    _, th = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    total_cnts = len(cnts)
    kept = 0
    best = None
    best_center = None
    best_area = 0.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < float(min_area) or area > float(max_area):
            continue
        kept += 1
        if area > best_area:
            best = c
            best_area = area
        cv2.drawContours(out, [c], -1, (0, 255, 0), 1)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    if best is not None:
        # Highlight the best blob (largest kept) like the detector would pick.
        cv2.drawContours(out, [best], -1, (255, 0, 255), 3)
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            best_center = (cx, cy)

    _draw_calibration_detection_marker(out, best_center, best_center is not None)

    cv2.putText(
        out,
        f"THRESH={thr}  MIN_AREA={min_area}  CNT={total_cnts}  KEPT={kept}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return _jpeg_response(out)


@app.get("/api/calibration/snapshot")
async def api_calibration_snapshot_state():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    count = len(getattr(state, "calib_snapshots", []))
    index = int(getattr(state, "calib_snapshot_index", -1))
    has_snapshot = count > 0 and 0 <= index < count and state.calib_frozen_frame is not None
    snapshots = []
    files = getattr(state, "calib_snapshot_files", [])
    timestamps = getattr(state, "calib_snapshot_ts", [])
    for i in range(count):
        file_path = files[i] if i < len(files) else None
        snapshots.append({
            "index": i,
            "ts": (timestamps[i] if i < len(timestamps) else None),
            "file": file_path,
            "name": (Path(file_path).name if file_path else None),
        })
    return JSONResponse({
        "mode": ("snapshot" if getattr(state, "calib_use_frozen", False) else "live"),
        "has_snapshot": has_snapshot,
        "snapshot_ts": getattr(state, "calib_frozen_ts", None),
        "snapshot_file": getattr(state, "calib_frozen_file", None),
        "snapshot_dir": str(getattr(state, "calib_snapshots_dir", "")),
        "snapshot_count": count,
        "snapshot_index": index,
        "snapshots": snapshots,
        "can_prev": has_snapshot and index > 0,
        "can_next": has_snapshot and index < (count - 1),
    })


def _save_calibration_snapshot_to_disk(frame: np.ndarray, ts: float, index: int) -> Optional[str]:
    if state is None:
        return None
    try:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        millis = int((ts - math.floor(ts)) * 1000)
        filename = f"calib_{stamp}_{millis:03d}_{index:04d}.jpg"
        out_path = state.calib_snapshots_dir / filename
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            return None
        return str(out_path)
    except Exception:
        return None

def _sync_calibration_snapshots_from_disk() -> None:
    if state is None:
        return
    try:
        disk_files = [p for p in sorted(state.calib_snapshots_dir.glob("*.jpg")) if p.is_file()]
    except Exception:
        return

    memory_files = [str(p) for p in getattr(state, "calib_snapshot_files", []) if p]
    disk_file_str = [str(p) for p in disk_files]
    if memory_files == disk_file_str and len(state.calib_snapshots) == len(disk_file_str):
        return

    snapshots: List[np.ndarray] = []
    timestamps: List[float] = []
    files: List[Optional[str]] = []
    for p in disk_files:
        try:
            img = cv2.imread(str(p))
            if img is None:
                continue
            snapshots.append(img)
            timestamps.append(float(p.stat().st_mtime))
            files.append(str(p))
        except Exception:
            continue

    state.calib_snapshots = snapshots
    state.calib_snapshot_ts = timestamps
    state.calib_snapshot_files = files

    if len(snapshots) <= 0:
        state.calib_snapshot_index = -1
        state.calib_frozen_frame = None
        state.calib_frozen_ts = None
        state.calib_frozen_file = None
        state.calib_use_frozen = False
        return

    if state.calib_snapshot_index < 0 or state.calib_snapshot_index >= len(snapshots):
        state.calib_snapshot_index = len(snapshots) - 1
    _set_calibration_snapshot_index(state.calib_snapshot_index)


@app.post("/api/calibration/capture")
async def api_calibration_capture():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    if state.last_raw_frame is None:
        return JSONResponse({"error": "no frame yet"}, status_code=409)
    snap = None
    try:
        snap = state.last_raw_frame.copy()
    except Exception:
        snap = state.last_raw_frame
    ts = time.time()
    state.calib_snapshots.append(snap)
    state.calib_snapshot_ts.append(ts)
    state.calib_snapshot_index = len(state.calib_snapshots) - 1
    snap_file = _save_calibration_snapshot_to_disk(snap, ts, state.calib_snapshot_index)
    state.calib_snapshot_files.append(snap_file)
    state.calib_frozen_frame = state.calib_snapshots[state.calib_snapshot_index]
    state.calib_frozen_ts = state.calib_snapshot_ts[state.calib_snapshot_index]
    state.calib_frozen_file = state.calib_snapshot_files[state.calib_snapshot_index]
    state.calib_use_frozen = True
    return JSONResponse({
        "ok": True,
        "mode": "snapshot",
        "snapshot_ts": state.calib_frozen_ts,
        "snapshot_file": state.calib_frozen_file,
        "snapshot_dir": str(state.calib_snapshots_dir),
        "snapshot_count": len(state.calib_snapshots),
        "snapshot_index": state.calib_snapshot_index,
    })


def _set_calibration_snapshot_index(snapshot_index: int) -> bool:
    if state is None:
        return False
    count = len(state.calib_snapshots)
    if count <= 0:
        state.calib_snapshot_index = -1
        state.calib_frozen_frame = None
        state.calib_frozen_ts = None
        state.calib_frozen_file = None
        return False
    if snapshot_index < 0:
        snapshot_index = 0
    if snapshot_index >= count:
        snapshot_index = count - 1
    state.calib_snapshot_index = snapshot_index
    state.calib_frozen_frame = state.calib_snapshots[snapshot_index]
    state.calib_frozen_ts = state.calib_snapshot_ts[snapshot_index]
    state.calib_frozen_file = state.calib_snapshot_files[snapshot_index]
    return True


@app.post("/api/calibration/snapshot/prev")
async def api_calibration_snapshot_prev():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    count = len(state.calib_snapshots)
    if count <= 0:
        return JSONResponse({"error": "no snapshots"}, status_code=409)
    idx = int(getattr(state, "calib_snapshot_index", -1))
    if idx <= 0:
        return JSONResponse({"error": "at first snapshot"}, status_code=409)
    if not _set_calibration_snapshot_index(idx - 1):
        return JSONResponse({"error": "no snapshots"}, status_code=409)
    state.calib_use_frozen = True
    return JSONResponse({
        "ok": True,
        "mode": "snapshot",
        "snapshot_ts": state.calib_frozen_ts,
        "snapshot_file": state.calib_frozen_file,
        "snapshot_dir": str(state.calib_snapshots_dir),
        "snapshot_count": len(state.calib_snapshots),
        "snapshot_index": state.calib_snapshot_index,
    })


@app.post("/api/calibration/snapshot/next")
async def api_calibration_snapshot_next():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    count = len(state.calib_snapshots)
    if count <= 0:
        return JSONResponse({"error": "no snapshots"}, status_code=409)
    idx = int(getattr(state, "calib_snapshot_index", -1))
    if idx >= (count - 1):
        return JSONResponse({"error": "at last snapshot"}, status_code=409)
    if not _set_calibration_snapshot_index(idx + 1):
        return JSONResponse({"error": "no snapshots"}, status_code=409)
    state.calib_use_frozen = True
    return JSONResponse({
        "ok": True,
        "mode": "snapshot",
        "snapshot_ts": state.calib_frozen_ts,
        "snapshot_file": state.calib_frozen_file,
        "snapshot_dir": str(state.calib_snapshots_dir),
        "snapshot_count": len(state.calib_snapshots),
        "snapshot_index": state.calib_snapshot_index,
    })


@app.post("/api/calibration/snapshot/select")
async def api_calibration_snapshot_select(payload: dict = Body(...)):
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    try:
        req_index = int(payload.get("index"))
    except Exception:
        return JSONResponse({"error": "invalid index"}, status_code=400)
    if not _set_calibration_snapshot_index(req_index):
        return JSONResponse({"error": "no snapshots"}, status_code=409)
    state.calib_use_frozen = True
    return JSONResponse({
        "ok": True,
        "mode": "snapshot",
        "snapshot_ts": state.calib_frozen_ts,
        "snapshot_file": state.calib_frozen_file,
        "snapshot_dir": str(state.calib_snapshots_dir),
        "snapshot_count": len(state.calib_snapshots),
        "snapshot_index": state.calib_snapshot_index,
    })


@app.post("/api/calibration/live")
async def api_calibration_live():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    state.calib_use_frozen = False
    return JSONResponse({
        "ok": True,
        "mode": "live",
        "has_snapshot": state.calib_frozen_frame is not None,
        "snapshot_file": state.calib_frozen_file,
        "snapshot_dir": str(getattr(state, "calib_snapshots_dir", "")),
        "snapshot_count": len(getattr(state, "calib_snapshots", [])),
        "snapshot_index": getattr(state, "calib_snapshot_index", -1),
    })


@app.post("/api/calibration/clear")
async def api_calibration_clear():
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    _sync_calibration_snapshots_from_disk()
    count = len(state.calib_snapshots)
    if count <= 0:
        return JSONResponse({"error": "no snapshots"}, status_code=409)

    idx = state.calib_snapshot_index
    if idx < 0 or idx >= count:
        idx = count - 1

    file_to_delete = state.calib_snapshot_files[idx] if idx < len(state.calib_snapshot_files) else None
    if file_to_delete:
        try:
            p = Path(file_to_delete)
            if p.exists():
                p.unlink()
        except Exception:
            pass

    state.calib_snapshots.pop(idx)
    state.calib_snapshot_ts.pop(idx)
    state.calib_snapshot_files.pop(idx)

    if len(state.calib_snapshots) <= 0:
        state.calib_snapshot_index = -1
        state.calib_frozen_frame = None
        state.calib_frozen_ts = None
        state.calib_frozen_file = None
        state.calib_use_frozen = False
        return JSONResponse({
            "ok": True,
            "mode": "live",
            "has_snapshot": False,
            "snapshot_file": None,
            "snapshot_dir": str(getattr(state, "calib_snapshots_dir", "")),
            "snapshot_count": 0,
            "snapshot_index": -1,
        })

    if idx >= len(state.calib_snapshots):
        idx = len(state.calib_snapshots) - 1
    _set_calibration_snapshot_index(idx)
    state.calib_use_frozen = True
    return JSONResponse({
        "ok": True,
        "mode": "snapshot",
        "has_snapshot": True,
        "snapshot_ts": state.calib_frozen_ts,
        "snapshot_file": state.calib_frozen_file,
        "snapshot_dir": str(getattr(state, "calib_snapshots_dir", "")),
        "snapshot_count": len(state.calib_snapshots),
        "snapshot_index": state.calib_snapshot_index,
    })


def _read_target_settings(cfg_path: Path) -> dict:
    cfg = parse_ini(cfg_path)
    return {
        "DistanceSimulated": cfg.distance_simulated,
        "DistanceReal": cfg.distance_real,
        "DistanceIsMetric": cfg.distance_is_metric,
        "VideoWidth": cfg.video_width,
        "VideoHeight": cfg.video_height,
        "Sensitivity": cfg.sensitivity,
        "Gain": cfg.gain,
        "RedSatMin": (cfg.red_sat_min if cfg.red_sat_min is not None else 70),
        "RedValMin": (cfg.red_val_min if cfg.red_val_min is not None else 70),
        "RedGateKernel": (cfg.red_gate_kernel if cfg.red_gate_kernel is not None else 7),
        "WhiteSatMax": (cfg.white_sat_max if cfg.white_sat_max is not None else 16),
        "WhiteValMin": (cfg.white_val_min if cfg.white_val_min is not None else 240),
        "Scale": cfg.scale,
        "PositionX": cfg.position_x,
        "PositionY": cfg.position_y,
        "TargetFilename": cfg.target_filename,
        "WeaponFilename": cfg.weapon_filename,
        "Shooter": cfg.shooter,
        "Rounds": cfg.rounds,
        "Timer": cfg.timer,
        "TimeLimit": cfg.time_limit,
        "Countdown": cfg.countdown,
        "HitLimit": cfg.hit_limit,
        "SLH": cfg.slh,
        "Prepare": cfg.prepare,
    }


def _save_settings_to_ini(cfg_path: Path, updates: dict) -> None:
    # create backup (best-effort)
    try:
        bak = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        shutil.copy(str(cfg_path), str(bak))
    except Exception:
        pass

    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    updated = set()
    in_target_section = False
    target_section_found = False

    for raw in lines:
        s = raw.strip()
        # detect Target settings section start
        if s.startswith("**") and "Target settings" in s:
            in_target_section = True
            target_section_found = True
            out_lines.append(raw)
            continue
        # leaving target section when encountering other section header
        if in_target_section and s.startswith("**") and "Target settings" not in s:
            in_target_section = False
        # try to update simple key=value lines
        if "=" in s and not s.startswith("**"):
            k, v = [x.strip() for x in s.split("=", 1)]
            if k in updates and updates[k] is not None:
                val = updates[k]
                out_lines.append(f"{k} = {val}")
                updated.add(k)
                continue
        out_lines.append(raw)

    # NOTE: this is intentionally a lightweight INI updater:
    # - keeps unknown lines as-is
    # - updates/creates key=value pairs
    # - best-effort backup to *.bak
    # If a key isn't found, we append it under a Target settings block.
    missing = {k: v for k, v in updates.items() if v is not None and k not in updated}
    if missing:
        if target_section_found:
            # insert before the first non-target section or at end; easiest is to append now
            out_lines.append("")
            out_lines.append("** Target settings **")
            for k, v in missing.items():
                out_lines.append(f"{k} = {v}")
        else:
            # no target section found: append a section block
            out_lines.append("")
            out_lines.append("** Target settings **")
            for k, v in missing.items():
                out_lines.append(f"{k} = {v}")

    cfg_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


@app.get("/api/config")
async def api_get_config():
    if state is None or state.config_path is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    return JSONResponse(_read_target_settings(state.config_path))


@app.post("/api/config")
async def api_set_config(payload: dict = Body(...)):
    if state is None or state.config_path is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    # Accept only known keys
    allowed = {
        "DistanceSimulated",
        "DistanceReal",
        "DistanceIsMetric",
        "VideoWidth",
        "VideoHeight",
        "Sensitivity",
        "Gain",
        "RedSatMin",
        "RedValMin",
        "RedGateKernel",
        "WhiteSatMax",
        "WhiteValMin",
        "Scale",
        "PositionX",
        "PositionY",
        "TargetFilename",
        "WeaponFilename",
        "Shooter",
        "Rounds",
        "Timer",
        "TimeLimit",
        "Countdown",
        "HitLimit",
        "SLH",
        "Prepare",
    }
    updates = {k: payload.get(k) for k in allowed if k in payload}
    # Normalize types for numeric fields
    def _num(v):
        try:
            return float(v)
        except Exception:
            return v
    for key in ("DistanceSimulated", "DistanceReal", "Scale", "PositionX", "PositionY", "VideoWidth", "VideoHeight"):
        if key in updates and updates[key] is not None:
            updates[key] = _num(updates[key])
    for key in ("Sensitivity", "Gain", "RedSatMin", "RedValMin", "RedGateKernel", "WhiteSatMax", "WhiteValMin"):
        if key in updates and updates[key] is not None:
            updates[key] = _num(updates[key])
    if "DistanceIsMetric" in updates and updates["DistanceIsMetric"] is not None:
        try:
            updates["DistanceIsMetric"] = int(float(updates["DistanceIsMetric"]))
        except Exception:
            pass
    # integer-like shooting settings
    for key in ("Rounds", "Timer", "TimeLimit", "Countdown", "HitLimit", "SLH", "Prepare", "RedSatMin", "RedValMin", "RedGateKernel", "WhiteSatMax", "WhiteValMin"):
        if key in updates and updates[key] is not None:
            try:
                updates[key] = int(float(updates[key]))
            except Exception:
                pass
    try:
        _save_settings_to_ini(state.config_path, updates)
    except Exception as ex:
        return JSONResponse({"error": f"config save failed: {ex}"}, status_code=500)
    # Apply scale/position to live target if provided
    try:
        cfg = parse_ini(state.config_path)
        if "Scale" in updates or "PositionX" in updates or "PositionY" in updates:
            _apply_scale_position(state, cfg.scale, cfg.position_x, cfg.position_y)
            _regenerate_from_last_raw()
    except Exception:
        pass
    try:
        return JSONResponse(_read_target_settings(state.config_path))
    except Exception as ex:
        return JSONResponse({"error": str(ex)}, status_code=500)


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@app.get("/api/targets")
async def api_targets_list():
    return JSONResponse({"targets": _list_data_files("targets", "*.tgt")})


@app.post("/api/targets/select")
async def api_targets_select(payload: dict = Body(...)):
    name = payload.get("name")
    if not name:
        return JSONResponse({"error": "missing name"}, status_code=400)
    available = _list_data_files("targets", "*.tgt")
    if name not in available:
        return JSONResponse({"error": "not found"}, status_code=404)
    global state
    state_updated = False
    if state:
        try:
            t = load_target(BASE.parent / "data" / "targets" / name)
            state.target = t
            state.active_target_filename = name
            # recompute scale/position using current frame size if available
            if state.frame_size:
                state.target.auto_scale(state.frame_size)
                # reset baseline to the new target's auto-scale
                state.initial_p2mm = state.target.info.pixel_to_mm_scale
                if state.base_center is None:
                    w, h = state.frame_size
                    state.base_center = (w / 2.0, h / 2.0)
                # Apply persisted Scale/PositionX/PositionY from ha.ini if present
                try:
                    if state.config_path and state.config_path.exists():
                        cfg = parse_ini(state.config_path)
                        _apply_scale_position(state, cfg.scale, cfg.position_x, cfg.position_y)
                except Exception:
                    pass
            state_updated = True
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    # regenerate overlay from last raw frame so clients see updated ghost immediately
    try:
        _regenerate_from_last_raw()
    except Exception:
        pass
    # persist selection to config file (use state's config_path if available, else default)
    cfg_path = (state.config_path if state and state.config_path else (BASE.parent / "config" / "ha.ini"))
    if not cfg_path.exists():
        # create minimal ini with target settings header
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("** Target settings **\n", encoding='utf-8')
    try:
        _save_settings_to_ini(cfg_path, {"TargetFilename": name})
    except Exception as ex:
        return JSONResponse({"error": f"config save failed: {ex}"}, status_code=500)
    return JSONResponse({"ok": True, "active": name, "state_updated": state_updated})


@app.get("/api/weapons")
async def api_weapons_list():
    return JSONResponse({"weapons": _list_data_files("weapons", "*.gun")})


@app.post("/api/weapons/select")
async def api_weapons_select(payload: dict = Body(...)):
    name = payload.get("name")
    if not name:
        return JSONResponse({"error": "missing name"}, status_code=400)
    available = _list_data_files("weapons", "*.gun")
    if name not in available:
        return JSONResponse({"error": "not found"}, status_code=404)
    global state
    state_updated = False
    if state:
        state.active_weapon_filename = name
        # try to parse weapon to update caliber if parser present
        try:
            cfg = parse_weapon(BASE.parent / "data" / "weapons" / name)
            if cfg and hasattr(cfg, "caliber_mm"):
                state.caliber_mm = cfg.caliber_mm
        except Exception:
            pass
        state_updated = True
    try:
        _regenerate_from_last_raw()
    except Exception:
        pass
    # persist selection to config file
    cfg_path = (state.config_path if state and state.config_path else (BASE.parent / "config" / "ha.ini"))
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("** Target settings **\n", encoding='utf-8')
    try:
        _save_settings_to_ini(cfg_path, {"WeaponFilename": name})
    except Exception as ex:
        return JSONResponse({"error": f"config save failed: {ex}"}, status_code=500)
    return JSONResponse({"ok": True, "active": name, "state_updated": state_updated})


@app.get("/api/weapons/{name}")
async def api_weapon_detail(name: str):
    available = _list_data_files("weapons", "*.gun")
    if name not in available:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        wp = parse_weapon(BASE.parent / "data" / "weapons" / name)
        return JSONResponse({
            "name": wp.name,
            "caliber": getattr(wp, "caliber", wp.caliber_mm),
            "caliber_mm": wp.caliber_mm,
            "unit": wp.unit,
        })
    except Exception as ex:
        return JSONResponse({"error": str(ex)}, status_code=500)


@app.post("/api/weapons/{name}")
async def api_weapon_update(name: str, payload: dict = Body(...)):
    available = _list_data_files("weapons", "*.gun")
    if name not in available:
        return JSONResponse({"error": "not found"}, status_code=404)
    new_name = payload.get("name", name)
    caliber = payload.get("caliber")
    unit = payload.get("unit", "mm")
    try:
        caliber_f = float(caliber) if caliber is not None else None
    except Exception:
        return JSONResponse({"error": "invalid caliber"}, status_code=400)
    if caliber_f is None:
        return JSONResponse({"error": "missing caliber"}, status_code=400)
    path = BASE.parent / "data" / "weapons" / name
    try:
        write_weapon(path, new_name, caliber_f, unit)
        # update active caliber if this is the selected weapon
        global state
        if state and state.active_weapon_filename == name:
            try:
                cfg = parse_weapon(path)
                state.caliber_mm = cfg.caliber_mm
                state.active_weapon_filename = name
            except Exception:
                pass
        return JSONResponse({"ok": True, "name": new_name, "unit": unit, "caliber": caliber_f})
    except Exception as ex:
        return JSONResponse({"error": str(ex)}, status_code=500)


# Startup: ensure background loop starts even if launched via uvicorn targetweb.server:app
@app.on_event("startup")
async def _startup_bg() -> None:
    global state
    # If already started by main, skip
    if state is not None:
        return
    try:
        # Read environment for configuration
        camera_index_env = os.environ.get("TARGETWEB_CAMERA_INDEX", "0")
        try:
            camera_index = int(camera_index_env)
        except Exception:
            camera_index = 0
        latest_dir = Path(os.environ.get("TARGETWEB_LATEST_DIR", str((BASE.parent / "run").resolve())))
        cfg_path = Path(os.environ.get("TARGETWEB_CONFIG", str((BASE.parent / "config" / "ha.ini").resolve())))
        target_env = os.environ.get("TARGETWEB_TARGET")
        caliber_env = os.environ.get("TARGETWEB_CALIBER", "4.5")
        device = os.environ.get("TARGETWEB_DEVICE")
        hid_enabled = os.environ.get("TARGETWEB_HID_ENABLED", "0") in ("1", "true", "True")
        hid_vid = os.environ.get("TARGETWEB_HID_VID")
        hid_pid = os.environ.get("TARGETWEB_HID_PID")

        caliber_mm = float(caliber_env)
        target_path: Optional[Path] = Path(target_env) if target_env else None
        cfg: Optional[AppConfig] = None
        if cfg_path and cfg_path.exists():
            cfg = parse_ini(cfg_path)
            if (target_path is None) and cfg.target_filename:
                target_path = (BASE.parent / "data" / "targets" / cfg.target_filename).resolve()
            if cfg.weapon_filename:
                wpath = (BASE.parent / "data" / "weapons" / cfg.weapon_filename).resolve()
                try:
                    caliber_mm = parse_weapon(wpath).caliber_mm
                except Exception:
                    pass
        if target_path is None:
            # default to a common included target
            target_path = (BASE.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
        tgt = load_target(target_path)

        hid_cfg = HIDConfig(
            enabled=hid_enabled,
            vendor_id=(int(hid_vid, 0) if hid_vid else None),
            product_id=(int(hid_pid, 0) if hid_pid else None),
        )

        from threading import Thread
        Thread(target=run_loop, kwargs=dict(
            camera_index=camera_index,
            tgt=tgt,
            caliber_mm=caliber_mm,
            latest_dir=latest_dir,
            device=device,
            app_cfg=cfg,
            cfg_path=cfg_path,
            hid_cfg=hid_cfg,
        ), daemon=True).start()
    except Exception as ex:
        # keep app up even if background fails; endpoints will serve placeholder
        print(f"Startup background failed: {ex}")


@app.post("/api/target/position")
async def api_target_position(dx: int = Body(...), dy: int = Body(...)):
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    cx, cy = state.target.info.position_px
    state.target.set_position(cx + dx, cy + dy)
    _save_scale_position_to_config(state)
    # immediately regenerate an updated image so UI and latest.jpg refresh
    _regenerate_from_last_raw()
    return JSONResponse({"ok": True, "position_px": state.target.info.position_px})


@app.post("/api/target/scale")
async def api_target_scale(factor: float = Body(..., embed=True)):
    if state is None:
        return JSONResponse({"error": "uninitialized"}, status_code=503)
    p2mm = max(state.target.info.pixel_to_mm_scale * factor, 0.05)
    state.target.set_scale(p2mm)
    # Only persist Scale when changing scale to avoid overwriting PositionX/Y
    _save_scale_only_to_config(state)
    # immediately regenerate an updated image so UI and latest.jpg refresh
    _regenerate_from_last_raw()
    return JSONResponse({"ok": True, "pixel_to_mm_scale": state.target.info.pixel_to_mm_scale})


# --- runner ---

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def run_loop(camera_index: int,
             tgt: ITarget,
             caliber_mm: float,
             latest_dir: Path,
             device: Optional[str],
             app_cfg: Optional[AppConfig],
             cfg_path: Optional[Path],
             hid_cfg: HIDConfig) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)
    # state will be initialized below; avoid referencing it before global assignment

    size = None
    if app_cfg and app_cfg.video_width and app_cfg.video_height:
        size = (int(app_cfg.video_width), int(app_cfg.video_height))
    src = USBCameraSource(index=int(camera_index), size=size or (640, 480), gain=(app_cfg.gain if app_cfg else None))

    # Detector tuning from HomeLESS-style settings
    def _detector_from_cfg(cfg: Optional[AppConfig]) -> BrightSpotDetector:
        # Sensitivity: higher => easier detection (lower threshold, lower min_area)
        s = 1.0
        if cfg and cfg.sensitivity is not None:
            try:
                s = float(cfg.sensitivity)
            except Exception:
                s = 1.0
        s = max(1.0, min(10.0, s))
        # Keep sensitivity=1 very strict to reduce glare false positives.
        threshold = int(round(252 - (s - 1.0) * 5.5))  # ~252..202
        # Adjust threshold based on Gain (Gain=10 ~ neutral). Higher gain -> higher threshold.
        if cfg and cfg.gain is not None:
            try:
                g = float(cfg.gain)
                threshold = int(round(threshold + (g - 10.0) * 0.8))
            except Exception:
                pass
        threshold = max(160, min(254, threshold))
        min_area = int(round(12 - (s - 1.0) * 1.0))    # ~12..3
        min_area = max(2, min(50, min_area))
        red_sat_min = 70
        red_val_min = 70
        red_gate_kernel = 7
        white_sat_max = 16
        white_val_min = 240
        if cfg is not None:
            try:
                if cfg.red_sat_min is not None:
                    red_sat_min = int(cfg.red_sat_min)
                if cfg.red_val_min is not None:
                    red_val_min = int(cfg.red_val_min)
                if cfg.red_gate_kernel is not None:
                    red_gate_kernel = int(cfg.red_gate_kernel)
                if cfg.white_sat_max is not None:
                    white_sat_max = int(cfg.white_sat_max)
                if cfg.white_val_min is not None:
                    white_val_min = int(cfg.white_val_min)
            except Exception:
                pass
        return BrightSpotDetector(
            threshold=threshold,
            min_area=min_area,
            red_sat_min=red_sat_min,
            red_val_min=red_val_min,
            red_gate_kernel=red_gate_kernel,
            white_sat_max=white_sat_max,
            white_val_min=white_val_min,
        )

    det = _detector_from_cfg(app_cfg)

    # optional Chromecast
    caster = None
    if device:
        try:
            import pychromecast
            chromecasts, _ = pychromecast.get_listed_chromecasts(friendly_names=[device])
            if chromecasts:
                caster = chromecasts[0]
                caster.wait()
                print(f"Connected to Chromecast: {caster.name}")
        except Exception as ex:
            print(f"Chromecast not available: {ex}")
            caster = None

    global state
    state = AppState(latest_dir=latest_dir, target=tgt, caliber_mm=caliber_mm, config_path=cfg_path)
    try:
        state.detector_threshold = int(getattr(det, "threshold", state.detector_threshold))
        state.detector_min_area = int(getattr(det, "min_area", state.detector_min_area))
        state.detector_red_sat_min = int(getattr(det, "red_sat_min", state.detector_red_sat_min))
        state.detector_red_val_min = int(getattr(det, "red_val_min", state.detector_red_val_min))
        state.detector_red_gate_kernel = int(getattr(det, "red_gate_kernel", state.detector_red_gate_kernel))
        state.detector_white_sat_max = int(getattr(det, "white_sat_max", state.detector_white_sat_max))
        state.detector_white_val_min = int(getattr(det, "white_val_min", state.detector_white_val_min))
    except Exception:
        pass

    # HID listener: on trigger, use last detected hit (only while shooting is running)
    def on_hid_trigger():
        if state is None:
            return
        with state.session_lock:
            # Match the UI's record gating: only count hits while running.
            if getattr(state, "shooting_status", "idle") != "running":
                return
            rs = getattr(state, "shooting_running_start_ts", None)
            if rs is not None and (time.time() - float(rs)) < SHOOTING_START_GRACE_SECONDS:
                return
            if state.last_hit is None:
                return
            h = state.last_hit
            s = tgt.get_points(h.x, h.y, caliber_mm)
            state.session.add(x=h.x, y=h.y, score=s)
            state.last_recorded_ts = time.time()
        print("HID trigger recorded")
    HIDListener(hid_cfg, on_hid_trigger).start()

    first_frame = True
    gain_alpha = None
    cfg_runtime = app_cfg
    cfg_mtime: Optional[float] = None
    if cfg_path is not None and cfg_path.exists():
        try:
            cfg_mtime = cfg_path.stat().st_mtime
        except Exception:
            cfg_mtime = None

    def _refresh_cfg_if_changed() -> None:
        nonlocal cfg_runtime, cfg_mtime, det, gain_alpha
        if cfg_path is None or (not cfg_path.exists()):
            return
        try:
            mt = cfg_path.stat().st_mtime
        except Exception:
            return
        if cfg_mtime is not None and mt == cfg_mtime:
            return
        cfg_mtime = mt
        try:
            cfg_runtime = parse_ini(cfg_path)
        except Exception:
            return
        det = _detector_from_cfg(cfg_runtime)
        try:
            state.detector_threshold = int(getattr(det, "threshold", state.detector_threshold))
            state.detector_min_area = int(getattr(det, "min_area", state.detector_min_area))
            state.detector_red_sat_min = int(getattr(det, "red_sat_min", state.detector_red_sat_min))
            state.detector_red_val_min = int(getattr(det, "red_val_min", state.detector_red_val_min))
            state.detector_red_gate_kernel = int(getattr(det, "red_gate_kernel", state.detector_red_gate_kernel))
            state.detector_white_sat_max = int(getattr(det, "white_sat_max", state.detector_white_sat_max))
            state.detector_white_val_min = int(getattr(det, "white_val_min", state.detector_white_val_min))
        except Exception:
            pass
        gain_alpha = None
        if cfg_runtime is not None and cfg_runtime.gain is not None:
            try:
                # Gain=10 => neutral (alpha=1.0)
                gain_alpha = max(0.1, float(cfg_runtime.gain) / 10.0)
            except Exception:
                gain_alpha = None

    # initial gain from config
    _refresh_cfg_if_changed()

    prev_hit_xy: Optional[Tuple[float, float]] = None
    stable_hit_frames = 0

    for frame in src.frames():
        _refresh_cfg_if_changed()
        now_ts = time.time()
        _shooting_tick(now_ts)
        if gain_alpha is not None:
            try:
                frame = cv2.convertScaleAbs(frame, alpha=gain_alpha, beta=0)
            except Exception:
                pass
        h, w = frame.shape[:2]
        state.last_raw_frame = frame
        if first_frame:
            tgt.auto_scale((w, h))
            tgt.set_position(w / 2.0, h / 2.0)
            # record baseline after auto-scale and centering
            state.initial_p2mm = tgt.info.pixel_to_mm_scale
            state.base_center = (w / 2.0, h / 2.0)
            state.frame_size = (w, h)
            # apply config scale/position if provided
            if app_cfg is not None:
                if app_cfg.scale and app_cfg.scale > 0:
                    tgt.set_scale(app_cfg.scale * tgt.info.pixel_to_mm_scale)
                # treat PositionX/Y as pixel offsets; ignore sentinel 5000
                px = app_cfg.position_x
                py = app_cfg.position_y
                if abs(px) < 4000 and abs(py) < 4000:
                    cx, cy = tgt.info.position_px
                    tgt.set_position(cx + px, cy + py)
            first_frame = False

        hit = det.detect(frame)
        # Reject very large blobs (usually glare) even if they pass contour area filtering.
        if hit is not None:
            try:
                if float(getattr(hit, "strength", 0.0)) > 25.0:
                    hit = None
            except Exception:
                pass
        # Use current selected caliber if user changes weapon while running.
        current_caliber_mm = caliber_mm
        try:
            if state is not None and getattr(state, "caliber_mm", None) is not None:
                current_caliber_mm = float(getattr(state, "caliber_mm"))
        except Exception:
            current_caliber_mm = caliber_mm
        score = None
        hit_xy = None
        if hit is not None:
            hit_xy = (hit.x, hit.y)
            score = tgt.get_points(hit.x, hit.y, current_caliber_mm)
            state.last_hit = hit
            if prev_hit_xy is None:
                stable_hit_frames = 1
            else:
                dx = float(hit.x) - float(prev_hit_xy[0])
                dy = float(hit.y) - float(prev_hit_xy[1])
                if (dx * dx + dy * dy) <= (HIT_CONFIRMATION_DISTANCE_PX * HIT_CONFIRMATION_DISTANCE_PX):
                    stable_hit_frames += 1
                else:
                    stable_hit_frames = 1
            prev_hit_xy = (float(hit.x), float(hit.y))
            # Auto-record hits only while shooting is running
            with state.session_lock:
                can_record = getattr(state, "shooting_status", "idle") == "running"
                running_start = getattr(state, "shooting_running_start_ts", None)
                record_armed = bool(getattr(state, "hit_record_armed", True))
                past_start_grace = (
                    running_start is not None
                    and (now_ts - float(running_start)) >= SHOOTING_START_GRACE_SECONDS
                )
                # simple debounce: avoid flooding on static frames or multi-frame glare
                if (
                    can_record
                    and past_start_grace
                    and record_armed
                    and stable_hit_frames >= HIT_CONFIRMATION_FRAMES
                    and (now_ts - float(getattr(state, "last_recorded_ts", 0.0))) >= RECORD_DEBOUNCE_SECONDS
                ):
                    state.session.add(x=hit.x, y=hit.y, score=score)
                    state.last_recorded_ts = now_ts
                    state.hit_record_armed = False
                    state.hit_nohit_frames = 0
                    stable_hit_frames = 0
        else:
            prev_hit_xy = None
            stable_hit_frames = 0
            with state.session_lock:
                miss_frames = int(getattr(state, "hit_nohit_frames", 0)) + 1
                state.hit_nohit_frames = miss_frames
                if miss_frames >= HIT_RELEASE_FRAMES:
                    state.hit_record_armed = True

        # Decide which recorded shots to draw (SLH = show last N hits).
        shots_to_draw: list[dict] = []
        try:
            with state.session_lock:
                raw = list(getattr(state.session, "shots", []) or [])
            n = 0
            try:
                n = int(getattr(cfg_runtime, "slh", 0) or 0)
            except Exception:
                n = 0
            if n > 0:
                start = max(0, len(raw) - n)
                shots_to_draw = [{"x": s.x, "y": s.y, "i": (i + 1)} for i, s in enumerate(raw[start:], start=start)]
            else:
                shots_to_draw = [{"x": s.x, "y": s.y, "i": (i + 1)} for i, s in enumerate(raw)]
        except Exception:
            shots_to_draw = []

        # overlay and save
        img = draw_overlay(
            frame,
            tgt.overlay_hint(),
            hit_xy,
            score,
            caliber_mm=current_caliber_mm,
            pixels_per_mm=getattr(tgt.info, "pixel_to_mm_scale", None),
            shots=shots_to_draw,
        )
        _save_rendered_image(img)
        # cleanup now handled in _save_rendered_image

        # Chromecast best-effort: refresh by replaying same URL with cache busting
        if caster is not None and state.last_frame_name is not None:
            try:
                mc = caster.media_controller
                # serve unique file path so Chromecast refreshes reliably
                url = f"http://{get_local_ip()}:{os.environ.get('PORT', '8000')}/frames/{state.last_frame_name}"
                mc.play_media(url, content_type="image/jpeg")
            except Exception as ex:
                print(f"Chromecast play error: {ex}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Laser target practice web app")
    parser.add_argument("--target", required=False, default=None, help="Path to .tgt file to use (optional; overrides ha.ini)")
    parser.add_argument("--caliber", type=float, default=4.5, help="Caliber in mm (e.g., 4.5 for .177)")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index (OpenCV VideoCapture index; usually 0)")
    parser.add_argument("--latest-dir", default=str((BASE.parent / "run").resolve()))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Chromecast friendly name (optional)")
    parser.add_argument("--config", default=str((BASE.parent / "config" / "ha.ini").resolve()), help="Path to a HomeLESS-compatible ha.ini config (local config/ha.ini)")
    parser.add_argument("--hid-enabled", action="store_true", help="Enable HID trigger listener")
    parser.add_argument("--hid-vid", type=lambda x: int(x, 0), default=None, help="HID Vendor ID (e.g., 0x1234)")
    parser.add_argument("--hid-pid", type=lambda x: int(x, 0), default=None, help="HID Product ID (e.g., 0x5678)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the browser")
    args = parser.parse_args()

    # resolve configuration: ha.ini and weapon
    caliber_mm = args.caliber
    target_path = Path(args.target) if args.target else None
    cfg_path = Path(args.config) if args.config else None
    cfg = None
    if cfg_path and cfg_path.exists():
        cfg = parse_ini(cfg_path)
        if (target_path is None) and cfg.target_filename:
            target_path = (BASE.parent / "data" / "targets" / cfg.target_filename).resolve()
        if cfg.weapon_filename:
            wpath = (BASE.parent / "data" / "weapons" / cfg.weapon_filename).resolve()
            try:
                caliber_mm = parse_weapon(wpath).caliber_mm
            except Exception:
                pass
    # if user passed a bare filename, resolve in local data store
    if target_path is not None:
        if (not target_path.is_absolute()) or (target_path.suffix.lower() != ".tgt"):
            candidate = (BASE.parent / "data" / "targets" / target_path.name).resolve()
            if candidate.exists():
                target_path = candidate

    if target_path is None:
        raise SystemExit("No target specified. Set TargetFilename in ha.ini or pass --target.")

    tgt = load_target(target_path)

    # Expose config to startup hook via environment and let startup launch the background loop
    os.environ["TARGETWEB_CAMERA_INDEX"] = str(int(args.camera_index))
    os.environ["TARGETWEB_LATEST_DIR"] = str(Path(args.latest_dir).resolve())
    os.environ["TARGETWEB_CONFIG"] = str(Path(args.config).resolve())
    if args.target:
        os.environ["TARGETWEB_TARGET"] = str(Path(args.target).resolve())
    os.environ["TARGETWEB_CALIBER"] = str(caliber_mm)
    if args.device:
        os.environ["TARGETWEB_DEVICE"] = args.device
    if args.hid_enabled:
        os.environ["TARGETWEB_HID_ENABLED"] = "1"
    if args.hid_vid is not None:
        os.environ["TARGETWEB_HID_VID"] = hex(args.hid_vid)
    if args.hid_pid is not None:
        os.environ["TARGETWEB_HID_PID"] = hex(args.hid_pid)

    # Auto-open default browser
    if not args.no_open:
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass

    uvicorn.run("targetweb.server:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
