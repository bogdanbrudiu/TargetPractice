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


def test_white_overexposed_corner_is_not_detected_without_red():
    # Simulate an overexposed white patch in a corner; detector should ignore it.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (70, 70), (255, 255, 255), -1)

    det = BrightSpotDetector(threshold=200, min_area=2)
    hit = det.detect(img)

    assert hit is None


def test_red_halo_with_white_core_is_detected():
    # Common laser bloom: white-hot center and a red surrounding halo.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(img, (200, 140), 5, (0, 0, 255), -1)        # red halo
    cv2.circle(img, (200, 140), 2, (255, 255, 255), -1)    # white core

    det = BrightSpotDetector(threshold=220, min_area=2)
    hit = det.detect(img)

    assert hit is not None
    assert abs(hit.x - 200) <= 3
    assert abs(hit.y - 140) <= 3


def test_red_spot_next_to_overexposed_bloom_is_detected():
    # Real laser bloom can have a red lobe plus an adjacent blown-out white patch.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(img, (180, 120), 4, (0, 0, 255), -1)
    cv2.circle(img, (186, 120), 6, (255, 255, 255), -1)

    det = BrightSpotDetector(threshold=220, min_area=2, red_gate_kernel=7)
    hit = det.detect(img)

    assert hit is not None
    assert abs(hit.x - 183) <= 5
    assert abs(hit.y - 120) <= 3


def test_dim_red_spot_on_black_background_is_detected_by_local_contrast_fallback():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(img, (150, 100), 4, (0, 0, 120), -1)

    det = BrightSpotDetector(threshold=220, min_area=2, red_gate_kernel=7)
    hit = det.detect(img)

    assert hit is not None
    assert abs(hit.x - 150) <= 4
    assert abs(hit.y - 100) <= 4


