import cv2
import numpy as np

from targetweb.detector import Hit
from targetweb.server import _hit_has_novel_brightening


def test_novel_brightening_accepts_new_spot():
    prev = np.zeros((120, 160, 3), dtype=np.uint8)
    curr = prev.copy()
    cv2.circle(curr, (80, 60), 4, (0, 0, 255), -1)

    assert _hit_has_novel_brightening(prev, curr, Hit(x=80.0, y=60.0, strength=4.0)) is True


def test_novel_brightening_rejects_persistent_static_spot():
    prev = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(prev, (80, 60), 4, (0, 0, 255), -1)
    curr = prev.copy()

    assert _hit_has_novel_brightening(prev, curr, Hit(x=80.0, y=60.0, strength=4.0)) is False