"""Clip checks, timing, and browser-ready H.264.

FFmpeg is used for normalisation, rotation, and playback. OpenCV reads the
normalised file. If presentation timestamps are missing or not increasing, the
clock falls back to frame index divided by the nominal frame rate and the
analysis records that fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from primed.config import Limits


class VideoError(ValueError):
    """A clip cannot be measured. The message is safe to show in the app."""


ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


@dataclass
class VideoInfo:
    path: Path
    filename: str
    sha256: str
    size_bytes: int
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str
    rotation_deg: float
    frame_count: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_detail(detail: str) -> str:
    """Turn an FFmpeg line into a message that does not include a local path."""
    lowered = detail.lower()
    if any(
        phrase in lowered
        for phrase in ("invalid data", "moov atom not found", "error opening input", "invalid argument")
    ):
        return "This file could not be decoded. Use an MP4, MOV, M4V, or WebM clip."
    cleaned = re.sub(r"(?:[A-Za-z]:)?(?:/|\\)[^\s:]+", "the clip", detail)
    return (cleaned or "The video command failed.")[:300]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoError("FFmpeg is not available on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or exc.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else "The video command failed."
        raise VideoError(_public_detail(detail)) from None


def probe(path: Path) -> dict:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoError("This file could not be decoded.") from exc


def _video_stream(payload: dict) -> dict:
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") == "video":
            return stream
    raise VideoError("This file has no video stream.")


def _fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        if float(denominator) == 0:
            return 0.0
        return float(numerator) / float(denominator)
    return float(value)


def rotation_degrees(stream: dict) -> float:
    """Read a rotate tag or a display-matrix angle, in degrees."""
    tags = stream.get("tags") or {}
    if tags.get("rotate") not in (None, ""):
        try:
            return float(tags["rotate"])
        except ValueError:
            pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return float(side["rotation"])
            except (TypeError, ValueError):
                continue
    return 0.0


def orientation_filter(rotation_deg: float) -> str:
    """FFmpeg filter that turns the stored frame into the displayed orientation.

    90° and -270° rotate clockwise (transpose=1). -90° and 270° rotate
    counter-clockwise (transpose=2). 180° flips both axes. Other angles are
    left unchanged rather than guessed.
    """
    quarter = int(round(rotation_deg)) % 360
    if quarter == 90:
        return "transpose=1"
    if quarter == 270:
        return "transpose=2"
    if quarter == 180:
        return "hflip,vflip"
    return ""


def validate_file(path: Path, filename: str, limits: Limits) -> VideoInfo:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise VideoError("Use an MP4, MOV, M4V, or WebM clip.")
    size = path.stat().st_size
    if size <= 0:
        raise VideoError("The file is empty.")
    if size > limits.max_bytes:
        raise VideoError(f"{filename} is larger than {limits.max_bytes // (1024 * 1024)} MB.")
    payload = probe(path)
    stream = _video_stream(payload)
    format_info = payload.get("format") or {}
    duration = float(stream.get("duration") or format_info.get("duration") or 0)
    if duration <= 0:
        raise VideoError("The clip duration could not be read.")
    if duration > limits.max_duration_s:
        raise VideoError(f"{filename} is longer than {limits.max_duration_s:.0f} seconds.")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width < 2 or height < 2:
        raise VideoError("The clip dimensions could not be read.")
    fps = _fraction(stream.get("avg_frame_rate")) or _fraction(stream.get("r_frame_rate"))
    if fps <= 1:
        fps = 30.0
    codec = stream.get("codec_name") or ""
    if not codec:
        raise VideoError("The video codec is not supported.")
    frames = stream.get("nb_frames")
    return VideoInfo(
        path=path,
        filename=Path(filename).name,
        sha256=sha256_file(path),
        size_bytes=size,
        duration_s=duration,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        rotation_deg=rotation_degrees(stream),
        frame_count=int(frames) if frames and str(frames).isdigit() else None,
    )


def normalize(src: Path, dest: Path, max_width: int, rotation_deg: float = 0) -> None:
    """Bake in rotation and scale so later frames match the displayed picture.

    ``-noautorotate`` keeps this from applying a display matrix twice when the
    caller has already supplied the angle.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    oriented = orientation_filter(rotation_deg)
    if oriented:
        filters.append(oriented)
    filters.append(f"scale='min({int(max_width)},iw)':-2")
    _run(
        [
            "ffmpeg",
            "-y",
            "-noautorotate",
            "-i",
            str(src),
            "-an",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )


def presentation_times(path: Path) -> tuple[list[float], str]:
    """Return per-frame timestamps. Fall back to a constant frame rate if needed."""
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    times: list[float] = []
    for line in (completed.stdout or "").splitlines():
        text = line.strip()
        if not text or text == "N/A":
            return [], "fps_fallback"
        try:
            times.append(float(text))
        except ValueError:
            return [], "fps_fallback"
    if len(times) < 2 or any(times[index] >= times[index + 1] for index in range(len(times) - 1)):
        return [], "fps_fallback"
    return times, "pts"


def monotonic_milliseconds(times: list[float]) -> list[int]:
    """MediaPipe VIDEO mode requires strictly increasing millisecond timestamps."""
    output: list[int] = []
    last = -1
    for moment in times:
        millis = int(round(moment * 1000))
        if millis <= last:
            millis = last + 1
        last = millis
        output.append(millis)
    return output


def sample_plan(duration_s: float, limits: Limits) -> float:
    """Lower the sample rate so a long clip stays inside the frame budget."""
    if duration_s <= 0:
        return limits.sample_fps
    budget = limits.max_frames / duration_s
    return min(limits.sample_fps, max(2.0, budget))


def iter_sampled_frames(path: Path, times: list[float], sample_fps: float, max_frames: int):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError("The normalised clip could not be read.")
    nominal = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if nominal <= 1:
        nominal = 30.0
    next_time = 0.0
    step = 1.0 / sample_fps
    index = 0
    produced = 0
    try:
        while produced < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if times and index < len(times):
                moment = times[index]
            elif times:
                moment = times[-1] + (index - len(times) + 1) / nominal
            else:
                moment = index / nominal
            if moment + 1e-6 >= next_time:
                yield index, float(moment), frame
                produced += 1
                next_time = moment + step
            index += 1
    finally:
        capture.release()


def transcode_h264(src: Path, dest: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    codec = _video_stream(probe(dest)).get("codec_name")
    if codec != "h264":
        raise VideoError("The playback file is not H.264.")


def extract_still(src: Path, dest: Path, at_seconds: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ss",
            f"{max(0.0, at_seconds):.3f}",
            "-frames:v",
            "1",
            str(dest),
        ]
    )


def extract_clip(src: Path, dest: Path, start_s: float, duration_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ss",
            f"{max(0.0, start_s):.3f}",
            "-t",
            f"{max(0.2, duration_s):.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )


def remove_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                for child in path.iterdir():
                    if child.is_file():
                        child.unlink()
        except OSError:
            continue
