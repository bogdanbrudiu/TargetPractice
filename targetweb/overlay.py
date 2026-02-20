from __future__ import annotations
from typing import Tuple, Iterable, Any
from PIL import Image, ImageDraw
import numpy as np


def draw_overlay(frame_bgr: np.ndarray,
                 hints: list[tuple[str, tuple[int, int, int, int]]],
                 hit_xy: tuple[float, float] | None,
                 score: float | None,
                 caliber_mm: float | None = None,
                 pixels_per_mm: float | None = None,
                 shots: Iterable[Any] | None = None) -> np.ndarray:
    # Convert to PIL RGBA for semi-transparent fills, then back to BGR
    frame_rgb = cv2_to_rgb(frame_bgr)
    base = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # draw target hints
    for kind, data in hints:
        if kind == "disc":
            cx, cy, r, alpha = data
            bbox = [cx - r, cy - r, cx + r, cy + r]
            # HomeLESS-like: semi-transparent yellow fill inside black area
            draw.ellipse(bbox, fill=(255, 255, 0, max(0, min(alpha, 255))))
        elif kind == "circle":
            cx, cy, r, thickness = data
            bbox = [cx - r, cy - r, cx + r, cy + r]
            for t in range(thickness):
                draw.ellipse([bbox[0] - t, bbox[1] - t, bbox[2] + t, bbox[3] + t], outline=(255, 255, 0, 255))
        elif kind == "rect":
            x, y, w, h = data
            draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0, 255), width=2)

    def _radius_px() -> int:
        r0 = 6
        try:
            if caliber_mm is not None and pixels_per_mm is not None:
                r0 = int(round((float(caliber_mm) * float(pixels_per_mm)) / 2.0))
        except Exception:
            r0 = 6
        return max(2, min(int(r0), 200))

    # Draw recorded shots first (so current live hit is on top)
    if shots is not None:
        r = _radius_px()
        try:
            for idx0, s in enumerate(shots):
                if s is None:
                    continue
                if isinstance(s, dict):
                    x = float(s.get("x"))
                    y = float(s.get("y"))
                    label = s.get("i")
                else:
                    x = float(getattr(s, "x"))
                    y = float(getattr(s, "y"))
                    label = None
                # semi-transparent fill + solid outline
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0, 60), outline=(255, 0, 0, 255), width=2)

                # Optional shot index label (matches Results row # when provided)
                try:
                    n = int(label) if label is not None else (idx0 + 1)
                    tx, ty = int(x + r + 2), int(y - r - 2)
                    # shadow then text for readability
                    draw.text((tx + 1, ty + 1), str(n), fill=(0, 0, 0, 200))
                    draw.text((tx, ty), str(n), fill=(255, 255, 255, 220))
                except Exception:
                    pass
        except Exception:
            pass

    if hit_xy is not None:
        x, y = hit_xy
        r = _radius_px()
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 255), width=2)

    if score is not None:
        text = f"Score: {score:.1f}" if isinstance(score, float) else f"Score: {score}"
        draw.text((10, 10), text, fill=(255, 255, 255, 255))
    # Composite overlay with transparency onto base
    composed = Image.alpha_composite(base, overlay)
    out = np.array(composed.convert("RGB"))
    return rgb_to_cv2(out)


def cv2_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_cv2(img_rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
