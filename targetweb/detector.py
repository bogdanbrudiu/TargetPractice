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

    def __init__(self, threshold: int = 230, min_area: int = 5, max_area_frac: float = 0.02):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area_frac = max_area_frac

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
        lower1 = (0, 70, 70)
        upper1 = (10, 255, 255)
        lower2 = (170, 70, 70)
        upper2 = (179, 255, 255)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def detect(self, frame: np.ndarray) -> Optional[Hit]:
        # Prefer red-laser detection first; it rejects white glare via saturation.
        try:
            red = self._hit_from_mask(self._red_mask(frame))
            if red is not None:
                return red
        except Exception:
            pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, int(self.threshold), 255, cv2.THRESH_BINARY)
        return self._hit_from_mask(th)
