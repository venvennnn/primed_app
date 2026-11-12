"""Run the pose pipeline once per clip and parameter set.

Widget reruns must call ``cached_analysis`` rather than ``analyze_upload``.
Changing the movement, view, facing, or segmentation version derives again
from the stored landmarks and does not repeat inference. ``force=True`` repeats it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from primed.config import POSE_MODEL_NAME, SEG_VERSION, Limits, limits_from_env, model_path
from primed.ids import new_id
from primed.measure import derive_analysis, series_from_landmarks
from primed.repository import LocalRepository
from primed.tracking import track_clip
from primed.video_io import (
    VideoError,
    normalize,
    presentation_times,
    probe,
    remove_files,
    sample_plan,
    validate_file,
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _model_sha() -> str:
    path = model_path()
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_key(video_sha: str, limits: Limits) -> str:
    return _hash(
        {
            "sha256": video_sha,
            "model": POSE_MODEL_NAME,
            "model_sha256": _model_sha(),
            "max_width": limits.max_width,
            "max_frames": limits.max_frames,
            "sample_fps": limits.sample_fps,
        }
    )


def result_key(pose: str, clip: dict, limits: Limits) -> str:
    return _hash(
        {
            "pose_key": pose,
            "profile_id": clip.get("profile_id"),
            "profile_version": "1",
            "seg_version": SEG_VERSION,
            "variant": clip.get("variant"),
            "view": clip.get("view"),
            "facing": clip.get("facing"),
            "handedness": clip.get("handedness"),
            "front_leg": (clip.get("context") or {}).get("front_leg"),
            "visibility_min": limits.visibility_min,
            "short_gap_s": limits.short_gap_s,
        }
    )


def find_analysis(workspace: dict, key: str) -> dict | None:
    for analysis in workspace.get("analyses", {}).values():
        if analysis.get("result_key") == key:
            return analysis
    return None


def ingest_upload(
    repository: LocalRepository,
    workspace: dict,
    file_bytes: bytes,
    filename: str,
    meta: dict,
    limits: Limits | None = None,
) -> dict:
    limits = limits or limits_from_env()
    if len(file_bytes) > limits.max_bytes:
        raise VideoError(f"{filename} is larger than {limits.max_bytes // (1024 * 1024)} MB.")
    clip_id = new_id("clip")
    folder = repository.clip_dir(clip_id)
    original = folder / "original.mp4"
    playback = folder / "playback.mp4"
    original.write_bytes(file_bytes)
    try:
        info = validate_file(original, filename, limits)
        normalize(original, playback, limits.max_width, info.rotation_deg)
    except Exception:
        remove_files([folder])
        try:
            folder.rmdir()
        except OSError:
            pass
        raise
    checked = probe(playback)
    stream = next(item for item in checked.get("streams", []) if item.get("codec_type") == "video")
    width = int(stream.get("width") or info.width)
    height = int(stream.get("height") or info.height)
    times, timing_mode = presentation_times(playback)
    duration = float((checked.get("format") or {}).get("duration") or info.duration_s)
    clip = {
        "id": clip_id,
        "filename": Path(filename).name,
        "sha256": info.sha256,
        "bytes": info.size_bytes,
        "duration_s": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(info.fps, 3),
        "rotation_deg": info.rotation_deg,
        "codec": info.codec,
        "timing_mode": timing_mode,
        "created_at": _now(),
        "recorded_at": meta.get("recorded_at") or _now(),
        "profile_id": meta["profile_id"],
        "variant": meta["variant"],
        "view": meta["view"],
        "facing": meta.get("facing") or "",
        "handedness": meta.get("handedness") or "",
        "context": dict(meta.get("context") or {}),
        "athlete": meta.get("athlete") or workspace.get("athlete") or "Athlete",
        "media_present": True,
        "status": "ready",
        "error": "",
        "warnings": [],
        "latest_analysis_id": "",
        "frame_times_available": timing_mode == "pts",
    }
    # Keep the timestamp list beside the clip so tracking can reuse it.
    (folder / "times.json").write_text(json.dumps({"mode": timing_mode, "times": times}))
    workspace.setdefault("clips", {})[clip_id] = clip
    return clip


def analyze_clip(
    repository: LocalRepository,
    workspace: dict,
    clip_id: str,
    *,
    force: bool = False,
    limits: Limits | None = None,
    progress=None,
) -> dict:
    limits = limits or limits_from_env()
    clip = workspace["clips"][clip_id]
    folder = repository.clip_dir(clip_id)
    key = pose_key(clip["sha256"], limits)
    derived_key = result_key(key, clip, limits)
    if not force:
        existing = find_analysis(workspace, derived_key)
        if existing is not None:
            clip["latest_analysis_id"] = existing["id"]
            clip["status"] = "ready"
            return existing
    cache_file = repository.cache_path(key)
    if cache_file.is_file() and not force:
        tracked = json.loads(cache_file.read_text())
    else:
        times_payload = json.loads((folder / "times.json").read_text())
        sample_fps = sample_plan(float(clip["duration_s"]), limits)
        tracked = track_clip(
            folder / "playback.mp4",
            times_payload.get("times") or [],
            times_payload.get("mode") or clip.get("timing_mode") or "fps_fallback",
            sample_fps,
            limits.max_frames,
            folder / "frames",
            folder / "overlay.mp4",
            min_visibility=limits.visibility_min,
            progress=progress,
        )
        cache_file.write_text(json.dumps({k: v for k, v in tracked.items() if k != "overlay"}))
    frames = tracked["frames"]
    series = series_from_landmarks(
        frames,
        int(tracked["width"]),
        int(tracked["height"]),
        clip.get("facing") or "",
        limits.visibility_min,
    )
    analysis = derive_analysis(
        profile_id=clip["profile_id"],
        times=[frame["t"] for frame in frames],
        series=series,
        landmarks=frames,
        meta={
            "clip_id": clip_id,
            "filename": clip["filename"],
            "variant": clip["variant"],
            "view": clip["view"],
            "facing": clip.get("facing") or "",
            "handedness": clip.get("handedness") or "",
            "context": clip.get("context") or {},
            "athlete": clip.get("athlete"),
            "recorded_at": clip.get("recorded_at"),
            "created_at": _now(),
            "synthetic": False,
            "source_kind": "upload",
            "width": tracked["width"],
            "height": tracked["height"],
            "timing_mode": tracked["timing_mode"],
            "fps": clip.get("fps"),
            "sample_fps": tracked["sample_fps"],
            "warnings": list(tracked.get("warnings") or []),
            "visibility_min": limits.visibility_min,
            "seg_version": SEG_VERSION,
        },
        short_gap_s=limits.short_gap_s,
    )
    analysis["pose_key"] = key
    analysis["result_key"] = derived_key
    analysis["model"] = tracked.get("model")
    workspace.setdefault("analyses", {})[analysis["id"]] = analysis
    clip["latest_analysis_id"] = analysis["id"]
    clip["status"] = "ready"
    clip["warnings"] = list(analysis.get("warnings") or [])
    clip["error"] = ""
    return analysis


def mark_failed(workspace: dict, clip_id: str | None, message: str) -> None:
    if clip_id and clip_id in workspace.get("clips", {}):
        workspace["clips"][clip_id]["status"] = "failed"
        workspace["clips"][clip_id]["error"] = message
