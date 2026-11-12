"""2D projected angles.

Angles use pixel coordinates so x and y share a unit. Normalised landmark
coordinates are not used directly, because a wide frame would stretch x
relative to y and change the angle.

For joints A, B and C the angle is at B, between vectors A−B and C−B.
A zero-length vector returns no angle. The cosine is clamped before arccos.

Trunk tilt is a side-view measurement. Image y grows downward. The torso
vector runs from the mid-hip to the mid-shoulder. Its signed angle from
image-up is positive when the shoulders sit to the right of the hips.

The reported tilt is then signed in the athlete's facing direction:

- facing "right" means the face points toward +x (the right of the frame)
- facing "left" means the face points toward −x

Positive trunk tilt means forward lean toward that facing direction.
Front and rear views do not use this sign; callers mark trunk tilt unavailable.
The result is a 2D projected angle in degrees, not a 3D joint angle.
"""

from __future__ import annotations

import math


def angle_deg(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float | None:
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosine = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def joint_angle_px(
    landmarks: list[list[float]] | None,
    width: int,
    height: int,
    i_a: int,
    i_b: int,
    i_c: int,
    min_visibility: float,
) -> float | None:
    """Angle at landmark i_b. Landmarks are normalised [x, y, visibility]."""
    if landmarks is None:
        return None
    points: list[tuple[float, float]] = []
    for index in (i_a, i_b, i_c):
        x, y, visibility = landmarks[index]
        if visibility < min_visibility:
            return None
        points.append((x * width, y * height))
    return angle_deg(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1])


def _midpoint(
    landmarks: list[list[float]],
    i_left: int,
    i_right: int,
    width: int,
    height: int,
    min_visibility: float,
) -> tuple[float, float] | None:
    lx, ly, lv = landmarks[i_left]
    rx, ry, rv = landmarks[i_right]
    if lv < min_visibility or rv < min_visibility:
        return None
    return ((lx + rx) * 0.5 * width, (ly + ry) * 0.5 * height)


def trunk_tilt_deg(
    landmarks: list[list[float]] | None,
    width: int,
    height: int,
    facing: str,
    min_visibility: float,
) -> float | None:
    """Positive means forward lean toward ``facing`` in a side view."""
    if landmarks is None or facing not in ("left", "right"):
        return None
    shoulder = _midpoint(landmarks, 11, 12, width, height, min_visibility)
    hip = _midpoint(landmarks, 23, 24, width, height, min_visibility)
    if shoulder is None or hip is None:
        return None
    torso_x = shoulder[0] - hip[0]
    torso_y = shoulder[1] - hip[1]
    if math.hypot(torso_x, torso_y) < 1e-6:
        return None
    # Angle from image-up (0, -1). Positive when shoulders are to the right.
    raw = math.degrees(math.atan2(torso_x, -torso_y))
    if facing == "right":
        return raw
    return -raw


def torso_length_px(
    landmarks: list[list[float]] | None,
    width: int,
    height: int,
    min_visibility: float,
) -> float | None:
    if landmarks is None:
        return None
    shoulder = _midpoint(landmarks, 11, 12, width, height, min_visibility)
    hip = _midpoint(landmarks, 23, 24, width, height, min_visibility)
    if shoulder is None or hip is None:
        return None
    length = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
    if length < 1e-6:
        return None
    return length


def hip_mid_y_px(
    landmarks: list[list[float]] | None,
    width: int,
    height: int,
    min_visibility: float,
) -> float | None:
    if landmarks is None:
        return None
    hip = _midpoint(landmarks, 23, 24, width, height, min_visibility)
    if hip is None:
        return None
    return hip[1]
