"""Runtime limits for the CPU demo.

Environment variables override the defaults. The upload cap stays at 24 MB so a
multipart request fits under Cloud Run's 32 MiB HTTP/1 limit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
POSE_MODEL_NAME = "pose_landmarker_lite"
PROFILE_VERSION = "1"
SEG_VERSION = "1"
EXPORT_VERSION = 1

# Short gaps may be bridged for a plotted line or for segmentation smoothing.
# Scalar measurements always use the raw samples.
SHORT_GAP_S = 0.20
VISIBILITY_MIN = 0.50
MIN_REFERENCE_REPS = 5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_path() -> Path:
    override = os.environ.get("PRIMED_MODEL_PATH")
    if override:
        return Path(override)
    return repo_root() / "models" / "pose_landmarker_lite.task"


def data_root() -> Path:
    override = os.environ.get("PRIMED_DATA_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "primed" / "sessions"


def gemini_model() -> str:
    return os.environ.get("PRIMED_GEMINI_MODEL", "gemini-3.8-flash")


def gemini_api_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 24 * 1024 * 1024
    max_duration_s: float = 45.0
    max_frames: int = 480
    sample_fps: float = 12.0
    max_width: int = 960
    min_reference_reps: int = MIN_REFERENCE_REPS
    short_gap_s: float = SHORT_GAP_S
    visibility_min: float = VISIBILITY_MIN


def limits_from_env() -> Limits:
    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        return default if raw is None or raw == "" else int(raw)

    def _float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        return default if raw is None or raw == "" else float(raw)

    return Limits(
        max_bytes=_int("PRIMED_MAX_BYTES", 24 * 1024 * 1024),
        max_duration_s=_float("PRIMED_MAX_DURATION_S", 45.0),
        max_frames=_int("PRIMED_MAX_FRAMES", 480),
        sample_fps=_float("PRIMED_SAMPLE_FPS", 12.0),
        max_width=_int("PRIMED_MAX_WIDTH", 960),
        min_reference_reps=_int("PRIMED_MIN_REFERENCE_REPS", MIN_REFERENCE_REPS),
        short_gap_s=_float("PRIMED_SHORT_GAP_S", SHORT_GAP_S),
        visibility_min=_float("PRIMED_VISIBILITY_MIN", VISIBILITY_MIN),
    )


def std_epsilon(unit: str) -> float:
    return {"deg": 0.05, "s": 0.005, "torso": 0.001, "rpm": 0.05}.get(unit, 1e-3)
