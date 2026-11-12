"""Angle convention tests."""

import math

from primed.geometry import angle_deg, joint_angle_px, trunk_tilt_deg


def test_right_angle_and_straight_line():
    assert angle_deg(0, 1, 0, 0, 1, 0) == 90
    straight = angle_deg(0, 0, 1, 0, 2, 0)
    assert straight == 180


def test_zero_length_vector_is_missing():
    assert angle_deg(0, 0, 0, 0, 1, 0) is None


def test_cosine_is_clamped_before_arccos(monkeypatch):
    real_hypot = math.hypot

    def slightly_short(*parts):
        return real_hypot(*parts) * 0.99

    # Shrink the vector lengths so the raw cosine would exceed 1. Without the
    # clamp, math.acos raises and the angle is lost.
    monkeypatch.setattr(math, "hypot", slightly_short)
    result = angle_deg(2, 0, 1, 0, 4, 0)
    assert result == 0


def test_pixel_angle_corrects_aspect_ratio():
    # 200 by 100 frame. Vectors that are not axis-aligned.
    # B (100, 50), A (100, 0), C (200, 100) -> about 116.57 degrees in pixels.
    width, height = 200, 100
    landmarks = [[0, 0, 0]] * 33
    landmarks[0] = [100 / width, 0 / height, 1]
    landmarks[1] = [100 / width, 50 / height, 1]
    landmarks[2] = [200 / width, 100 / height, 1]
    measured = joint_angle_px(landmarks, width, height, 0, 1, 2, 0.5)
    assert measured == math.degrees(math.acos(-2500 / (50 * math.hypot(100, 50))))
    assert abs(measured - 116.565) < 0.01


def test_low_visibility_is_missing():
    landmarks = [[0.5, 0.2, 1], [0.5, 0.5, 0.1], [0.8, 0.5, 1]] + [[0, 0, 0]] * 30
    assert joint_angle_px(landmarks, 100, 100, 0, 1, 2, 0.5) is None


def test_trunk_tilt_sign_follows_facing():
    width = height = 200
    landmarks = [[0, 0, 0]] * 33
    # Shoulders to the right of the hips, both shoulders and hips visible.
    landmarks[11] = [0.62, 0.30, 1]
    landmarks[12] = [0.70, 0.30, 1]
    landmarks[23] = [0.48, 0.60, 1]
    landmarks[24] = [0.52, 0.60, 1]
    facing_right = trunk_tilt_deg(landmarks, width, height, "right", 0.5)
    facing_left = trunk_tilt_deg(landmarks, width, height, "left", 0.5)
    assert facing_right is not None and facing_right > 0
    assert facing_left is not None and facing_left < 0
    assert abs(facing_right + facing_left) < 1e-6


def test_upright_trunk_is_near_zero():
    width = height = 200
    landmarks = [[0, 0, 0]] * 33
    landmarks[11] = [0.45, 0.30, 1]
    landmarks[12] = [0.55, 0.30, 1]
    landmarks[23] = [0.45, 0.60, 1]
    landmarks[24] = [0.55, 0.60, 1]
    tilt = trunk_tilt_deg(landmarks, width, height, "right", 0.5)
    assert abs(tilt) < 1e-6


def test_front_view_has_no_signed_tilt():
    landmarks = [[0.5, 0.3, 1]] * 33
    assert trunk_tilt_deg(landmarks, 100, 100, "front", 0.5) is None
