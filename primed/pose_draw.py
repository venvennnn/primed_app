"""Draw a 2D pose. Tracking gaps stay visible instead of being filled in."""

from __future__ import annotations

import math

POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
)
BODY_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
MEASURED_COLOR = (133, 150, 40)  # BGR teal
REFERENCE_COLOR = (90, 170, 70)  # BGR green


def draw_pose(frame, landmarks: list[list[float]] | None, min_visibility: float = 0.5):
    import cv2

    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if not landmarks:
        _banner(canvas, "Tracking gap")
        return canvas
    visible = []
    for index in range(min(33, len(landmarks))):
        x, y, visibility = landmarks[index]
        if visibility < min_visibility:
            visible.append(None)
            continue
        visible.append((int(round(x * width)), int(round(y * height))))
    drawn = False
    for start, end in POSE_CONNECTIONS:
        if start < len(visible) and end < len(visible) and visible[start] and visible[end]:
            cv2.line(canvas, visible[start], visible[end], MEASURED_COLOR, 2, cv2.LINE_AA)
            drawn = True
    for point in visible:
        if point is not None:
            cv2.circle(canvas, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 3, MEASURED_COLOR, 1, cv2.LINE_AA)
            drawn = True
    if not drawn:
        _banner(canvas, "Tracking gap")
    return canvas


def _banner(frame, text: str) -> None:
    import cv2

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 36), (width, height), (16, 24, 49), -1)
    cv2.putText(frame, text, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (243, 246, 253), 1, cv2.LINE_AA)


def normalise_landmarks(landmarks: list[list[float]] | None, width: int, height: int, min_visibility: float = 0.5):
    """Hip-centred pose in units of torso length. None when the torso is not visible."""
    if not landmarks:
        return None

    def point(index: int):
        x, y, visibility = landmarks[index]
        if visibility < min_visibility:
            return None
        return x * width, y * height

    left_hip, right_hip = point(23), point(24)
    left_shoulder, right_shoulder = point(11), point(12)
    if not all((left_hip, right_hip, left_shoulder, right_shoulder)):
        return None
    hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    scale = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
    if scale < 8:
        return None
    points = {}
    for index in BODY_INDICES:
        item = point(index)
        if item is None:
            continue
        points[str(index)] = [round((item[0] - hip[0]) / scale, 4), round((item[1] - hip[1]) / scale, 4)]
    return {"hip": hip, "scale": scale, "points": points}


def draw_reference_shape(frame, current_landmarks, reference_points: dict | None, min_visibility: float = 0.5):
    """Draw the reference shape in the current torso frame. Shape only, not a shared clock."""
    import cv2

    if not reference_points:
        return frame
    height, width = frame.shape[:2]
    current = normalise_landmarks(current_landmarks, width, height, min_visibility)
    if current is None:
        return frame
    hip = current["hip"]
    scale = current["scale"]
    located = {}
    for key, value in reference_points.items():
        located[int(key)] = (int(round(hip[0] + value[0] * scale)), int(round(hip[1] + value[1] * scale)))
    for start, end in POSE_CONNECTIONS:
        if start in located and end in located:
            cv2.line(frame, located[start], located[end], REFERENCE_COLOR, 2, cv2.LINE_AA)
    for point in located.values():
        cv2.circle(frame, point, 3, REFERENCE_COLOR, -1, cv2.LINE_AA)
    return frame
