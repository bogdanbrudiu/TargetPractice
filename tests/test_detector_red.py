import numpy as np
import cv2

from targetweb.detector import BrightSpotDetector


def test_red_laser_spot_detected_on_white_background():
    # White background with a small red dot.
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.circle(img, (160, 120), 3, (0, 0, 255), -1)  # BGR red

    det = BrightSpotDetector(threshold=245, min_area=2)
    hit = det.detect(img)

    assert hit is not None
    assert abs(hit.x - 160) <= 2
    assert abs(hit.y - 120) <= 2
