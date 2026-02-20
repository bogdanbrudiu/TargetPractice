from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TargetInfo:
    filename: str
    display_name: str
    type: int  # 0/2 circle (metric/imperial), 4 monoscope
    is_metric: bool
    resolution: Tuple[int, int] = (640, 480)
    # Pixel scaling: pixels per millimeter (derived per-frame)
    pixel_to_mm_scale: float = 1.0
    # Current target position in pixels (center)
    position_px: Tuple[float, float] = (320.0, 240.0)


@dataclass
class DiameterPoints:
    diameter_mm: float
    points: int


class ITarget:
    def __init__(self, info: TargetInfo):
        self.info = info

    def auto_scale(self, frame_size: Tuple[int, int]) -> None:
        raise NotImplementedError

    def set_scale(self, pixel_to_mm_scale: float) -> None:
        self.info.pixel_to_mm_scale = pixel_to_mm_scale

    def set_position(self, x_px: float, y_px: float) -> None:
        self.info.position_px = (x_px, y_px)

    def get_points(self, x_px: float, y_px: float, caliber_mm: float) -> float:
        raise NotImplementedError

    def overlay_hint(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Returns list of drawable hints, e.g. ("circle", (cx, cy, radius_px, thickness)).
        Used by UI overlay to draw ghost image.
        """
        return []


class CircleTarget(ITarget):
    def __init__(self, info: TargetInfo, rings: List[DiameterPoints], marked_index: int):
        super().__init__(info)
        # rings[0] corresponds to 1 point; rings[-1] highest points (10)
        self.rings = rings
        self.marked_index = marked_index  # zero-based index of black ring
        # derived (in pixels) after scale
        self._radius_px: List[float] = [0.0 for _ in self.rings]

    @property
    def max_diameter_mm(self) -> float:
        return self.rings[0].diameter_mm if self.rings else 0.0

    def auto_scale(self, frame_size: Tuple[int, int]) -> None:
        # mimic Processing: scale to fit height with 100px margin based on biggest diameter
        h = frame_size[1]
        max_d_mm = self.rings[0].diameter_mm
        if max_d_mm <= 0:
            self.set_scale(1.0)
        else:
            scale = max((h - 100) / max_d_mm, 0.1)
            self.set_scale(scale)
        self._recompute_radii()

    def set_scale(self, pixel_to_mm_scale: float) -> None:
        super().set_scale(pixel_to_mm_scale)
        self._recompute_radii()

    def _recompute_radii(self) -> None:
        p2mm = self.info.pixel_to_mm_scale
        self._radius_px = [(ring.diameter_mm * p2mm) / 2.0 for ring in self.rings]

    def get_points(self, x_px: float, y_px: float, caliber_mm: float) -> float:
        cx, cy = self.info.position_px
        dx, dy = (cx - x_px), (cy - y_px)
        # subtract half-caliber (in pixels)
        r = (dx * dx + dy * dy) ** 0.5 - (caliber_mm * self.info.pixel_to_mm_scale) / 2.0
        if r < 0:
            # center overlayed by pellet => max points
            return self.rings[-1].points
        if r > self._radius_px[0]:
            return 0.0
        points = self.rings[0].points
        # walk inward radius thresholds
        for i in range(1, len(self._radius_px)):
            if r > self._radius_px[i]:
                break
            points = self.rings[i].points
        return float(points)

    def overlay_hint(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        cx, cy = map(int, self.info.position_px)
        hints: List[Tuple[str, Tuple[int, int, int, int]]] = []
        # HomeLESS-like fills: outer area (lighter), inner black area (stronger)
        if self._radius_px:
            # Outer up to maximum ring with lower opacity
            hints.append(("disc", (cx, cy, int(self._radius_px[0]), 60)))
        if 0 <= self.marked_index < len(self._radius_px):
            # Inner black (marked) area with stronger opacity
            hints.append(("disc", (cx, cy, int(self._radius_px[self.marked_index]), 140)))
        # draw all rings (thin)
        for r in self._radius_px:
            hints.append(("circle", (cx, cy, int(r), 1)))
        # emphasize marked (black) ring and the outer ring
        if self._radius_px:
            max_r = int(self._radius_px[0])
            hints.append(("circle", (cx, cy, max_r, 2)))
        if 0 <= self.marked_index < len(self._radius_px):
            hints.append(("circle", (cx, cy, int(self._radius_px[self.marked_index]), 2)))
        return hints


class MonoscopeTarget(ITarget):
    def __init__(self, info: TargetInfo, width_px: int, height_px: int):
        super().__init__(info)
        # monoscope uses pixel resolution directly
        self.mono_w = width_px
        self.mono_h = height_px
        self.info.resolution = (width_px, height_px)

    def auto_scale(self, frame_size: Tuple[int, int]) -> None:
        # Monoscope is defined in raw pixels; use scale 1.0 to match its native resolution.
        self.set_scale(1.0)

    def get_points(self, x_px: float, y_px: float, caliber_mm: float) -> float:
        cx, cy = self.info.position_px
        w = self.mono_w * self.info.pixel_to_mm_scale
        h = self.mono_h * self.info.pixel_to_mm_scale
        left = cx - w / 2
        right = cx + w / 2
        top = cy - h / 2
        bottom = cy + h / 2
        return 1.0 if (left <= x_px <= right and top <= y_px <= bottom) else 0.0

    def overlay_hint(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        cx, cy = self.info.position_px
        w = int(self.mono_w * self.info.pixel_to_mm_scale)
        h = int(self.mono_h * self.info.pixel_to_mm_scale)
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return [("rect", (x, y, w, h))]


def load_target(path: Path) -> ITarget:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        raise ValueError(f"Invalid target file: {path}")
    name = lines[0]
    t_type = int(lines[1])
    info = TargetInfo(filename=str(path.name), display_name=name, type=t_type, is_metric=(t_type not in (2, 3)))

    if t_type in (0, 2):  # circle metric/imperial
        if len(lines) < 13:
            raise ValueError(f"Circle target needs 10 diameters + marker: {path}")
        # per HomeLESS: 10 diameters, then marked index (1-based)
        rings: List[DiameterPoints] = []
        for i in range(10):
            d = float(lines[2 + i].replace(",", "."))
            # convert from inches to mm if imperial
            if t_type == 2:
                d *= 25.4
            rings.append(DiameterPoints(diameter_mm=d, points=i + 1))
        marked = max(min(int(lines[12]) - 1, 9), 0)
        return CircleTarget(info, rings, marked)

    if t_type == 4:  # monoscope
        if len(lines) < 4:
            raise ValueError(f"Monoscope target needs width/height: {path}")
        w = int(lines[2])
        h = int(lines[3])
        return MonoscopeTarget(info, w, h)

    raise ValueError(f"Unsupported target type {t_type} in {path}")
