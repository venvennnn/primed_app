"""MediaPipe Pose Landmarker in VIDEO mode.

A new landmarker is created for every clip and closed before the next one.
Timestamps passed to the tracker are strictly increasing milliseconds.
"""

from __future__ import annotations

from pathlib import Path

from primed.config import POSE_MODEL_NAME, VISIBILITY_MIN, model_path
from primed.measure import landmarks_from_result
from primed.pose_draw import draw_pose
from primed.video_io import VideoError, iter_sampled_frames, monotonic_milliseconds, transcode_h264


class ModelMissing(VideoError):
    pass


def require_model() -> Path:
    path = model_path()
    if not path.is_file():
        raise ModelMissing(
            "The pose model is not installed. Run python scripts/fetch_pose_model.py and try again."
        )
    return path


def track_clip(
    video_path: Path,
    frame_times: list[float],
    timing_mode: str,
    sample_fps: float,
    max_frames: int,
    frames_dir: Path,
    overlay_path: Path,
    *,
    min_visibility: float = VISIBILITY_MIN,
    progress=None,
) -> dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    model = require_model()
    frames_dir.mkdir(parents=True, exist_ok=True)
    sampled = list(iter_sampled_frames(video_path, frame_times, sample_fps, max_frames))
    if not sampled:
        raise VideoError("No frames could be read from this clip.")
    height, width = sampled[0][2].shape[:2]
    millis = monotonic_milliseconds([moment for _index, moment, _frame in sampled])
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    intermediate = overlay_path.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(sample_fps, 1.0),
        (width, height),
    )
    payload_frames = []
    warnings = []
    if timing_mode == "fps_fallback":
        warnings.append("Exact presentation timestamps were unavailable, so frame time uses the nominal frame rate.")
    if len(sampled) >= max_frames:
        warnings.append(f"Analysis stopped at the {max_frames}-frame budget.")
    try:
        for ordinal, ((index, moment, frame), timestamp_ms) in enumerate(zip(sampled, millis)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
            landmarks = landmarks_from_result(result)
            drawn = draw_pose(frame, landmarks, min_visibility)
            if writer.isOpened():
                writer.write(drawn)
            cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            payload_frames.append({"index": int(index), "t": round(float(moment), 4), "lm": landmarks})
            if progress is not None:
                progress((ordinal + 1) / len(sampled), f"Tracking frame {ordinal + 1} of {len(sampled)}")
    finally:
        landmarker.close()
        writer.release()
    overlay_ready = False
    if intermediate.is_file() and intermediate.stat().st_size > 0:
        try:
            transcode_h264(intermediate, overlay_path)
            overlay_ready = True
        except VideoError as exc:
            warnings.append(f"Pose playback could not be encoded ({exc}). Exact frames are still available.")
        finally:
            intermediate.unlink(missing_ok=True)
    else:
        warnings.append("Pose playback could not be encoded. Exact frames are still available.")
        intermediate.unlink(missing_ok=True)
    tracked = sum(1 for frame in payload_frames if frame["lm"])
    if tracked == 0:
        warnings.append("No pose was detected. Measurements are empty rather than estimated.")
    return {
        "model": POSE_MODEL_NAME,
        "width": width,
        "height": height,
        "timing_mode": timing_mode,
        "sample_fps": sample_fps,
        "frames": payload_frames,
        "warnings": warnings,
        "overlay": overlay_ready,
        "tracking_fraction": round(tracked / len(payload_frames), 4) if payload_frames else 0.0,
    }
