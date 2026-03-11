from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple
import cv2
import numpy as np


@dataclass
class Hit:
    x: float
    y: float
    strength: float


class FrameSource:
    def frames(self) -> Iterable[np.ndarray]:
        raise NotImplementedError


class USBCameraSource(FrameSource):
    def __init__(self, index: int = 0, size: Tuple[int, int] = (640, 480), gain: Optional[float] = None):
        self.index = index
        self.size = size
        self.gain = gain

    def frames(self) -> Iterable[np.ndarray]:
        cap = cv2.VideoCapture(self.index)
        if self.size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if self.gain is not None:
            try:
                cap.set(cv2.CAP_PROP_GAIN, float(self.gain))
            except Exception:
                pass
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


class BrightSpotDetector:
    """
    Simple bright-spot detector:
    - grayscale, blur
    - high threshold
    - largest contour center as hit
    This is NOT using LaserGunTargetCaster's method; it's independent for prototyping.
    """

    def __init__(
        self,
        threshold: int = 230,
        min_area: int = 5,
        max_area_frac: float = 0.02,
        red_sat_min: int = 70,
        red_val_min: int = 70,
        red_gate_kernel: int = 7,
        white_sat_max: int = 16,
        white_val_min: int = 240,
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.red_sat_min = int(max(0, min(255, red_sat_min)))
        self.red_val_min = int(max(0, min(255, red_val_min)))
        k = int(max(1, red_gate_kernel))
        if (k % 2) == 0:
            k += 1
        self.red_gate_kernel = k
        self.white_sat_max = int(max(0, min(255, white_sat_max)))
        self.white_val_min = int(max(0, min(255, white_val_min)))

    def _threshold_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, int(self.threshold), 255, cv2.THRESH_BINARY)
        return th

    def _hit_from_mask(self, mask: np.ndarray) -> Optional[Hit]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        img_area = float(mask.shape[0] * mask.shape[1])
        max_area = max(self.min_area, img_area * float(self.max_area_frac))
        # Prefer the largest blob but ignore huge overexposed regions.
        best = None
        best_area = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < float(self.min_area) or area > float(max_area):
                continue
            if area > best_area:
                best = c
                best_area = area
        if best is None:
            return None
        (x, y), r = cv2.minEnclosingCircle(best)
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x, y
        return Hit(x=float(cx), y=float(cy), strength=float(r))

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary mask for red-ish pixels.

        Uses HSV thresholds so overexposed white areas (low saturation) are rejected.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # OpenCV hue: 0..179
        sat = int(self.red_sat_min)
        val = int(self.red_val_min)
        lower1 = (0, sat, val)
        upper1 = (10, 255, 255)
        lower2 = (170, sat, val)
        upper2 = (179, 255, 255)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _hybrid_mask(self, frame: np.ndarray) -> np.ndarray:
        # Require intensity near red hue/saturation to suppress white glare.
        # A red pointer often has a white-hot center with a red halo, so we
        # dilate the red mask before intersecting with threshold.
        th = self._threshold_mask(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = self._red_mask(frame)
        k = int(self.red_gate_kernel)
        red_gate = cv2.dilate(red, np.ones((k, k), np.uint8), iterations=1)
        # White-core assist: catches overexposed laser cores on white paper.
        white_base = cv2.inRange(
            hsv,
            (0, 0, int(self.white_val_min)),
            (179, int(self.white_sat_max), 255),
        )
        # Keep only local bright peaks, not the whole white paper background.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
        _, white_local = cv2.threshold(top_hat, 12, 255, cv2.THRESH_BINARY)
        white = cv2.bitwise_and(white_base, white_local)
        color_gate = cv2.bitwise_or(red_gate, white)
        mask = cv2.bitwise_and(th, color_gate)
        # Mild cleanup only; avoid eroding tiny laser spots away.
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def _contour_is_red_dominant(self, frame: np.ndarray, contour: np.ndarray) -> bool:
        # Guard threshold-only candidates with a red dominance check.
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return False
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return False

        local = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(local, [shifted], -1, 255, thickness=-1)

        mean_bgr = cv2.mean(roi, mask=local)
        b = float(mean_bgr[0])
        g = float(mean_bgr[1])
        r = float(mean_bgr[2])
        rg_max = max(g, b)
        return (r > 1.20 * rg_max) and ((r - rg_max) >= 20.0)

    def _hit_from_threshold_red_gated(self, frame: np.ndarray, threshold_mask: np.ndarray) -> Optional[Hit]:
        cnts, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        img_area = float(threshold_mask.shape[0] * threshold_mask.shape[1])
        max_area = max(self.min_area, img_area * float(self.max_area_frac))

        best = None
        best_area = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < float(self.min_area) or area > float(max_area):
                continue
            if not self._contour_is_red_dominant(frame, c):
                continue
            if area > best_area:
                best = c
                best_area = area
        if best is None:
            return None

        (x, y), r = cv2.minEnclosingCircle(best)
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x, y
        return Hit(x=float(cx), y=float(cy), strength=float(r))

    def detect(self, frame: np.ndarray) -> Optional[Hit]:
        # Use hybrid only so runtime detection matches calibration "hybrid" view.
        try:
            return self._hit_from_mask(self._hybrid_mask(frame))
        except Exception:
            return None
