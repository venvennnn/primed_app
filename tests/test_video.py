"""Video checks and a real pose pass on a clip with nobody in it."""

from pathlib import Path

import pytest

from primed.analyze import analyze_clip, ingest_upload
from primed.config import Limits
from primed.pose_draw import draw_pose
from primed.repository import CloudNotConfigured, LocalRepository, open_repository
from primed.video_io import VideoError, monotonic_milliseconds, transcode_h264, validate_file
from primed.workspace import new_workspace


def _color_clip(path: Path, duration: float = 0.4, size: str = "320x240", extra: list[str] | None = None) -> None:
    import subprocess

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s={size}:d={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    if extra:
        command = command[:1] + extra + command[1:]
    subprocess.run(command, check=True, capture_output=True)


def test_rejects_a_text_file(tmp_path: Path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    with pytest.raises(VideoError, match="could not be decoded") as caught:
        validate_file(path, "notes.mp4", Limits())
    assert str(tmp_path) not in str(caught.value)


def test_rejects_duration_and_size(tmp_path: Path):
    path = tmp_path / "short.mp4"
    _color_clip(path, duration=0.4)
    with pytest.raises(VideoError, match="longer"):
        validate_file(path, "short.mp4", Limits(max_duration_s=0.1))
    with pytest.raises(VideoError, match="larger"):
        validate_file(path, "short.mp4", Limits(max_bytes=100))


def test_accepts_a_short_clip_and_reports_timing(tmp_path: Path):
    path = tmp_path / "short.mp4"
    _color_clip(path, duration=0.5)
    info = validate_file(path, "short.mp4", Limits())
    assert info.duration_s > 0
    assert info.width == 320
    assert info.codec == "h264"


def test_rotation_metadata_is_read_and_baked_in(tmp_path: Path):
    from primed.video_io import normalize, probe, rotation_degrees

    assert rotation_degrees({"tags": {"rotate": "90"}}) == 90
    assert rotation_degrees({"side_data_list": [{"rotation": -90}]}) == -90
    source = tmp_path / "wide.mp4"
    _color_clip(source, duration=0.3)
    output = tmp_path / "portrait.mp4"
    normalize(source, output, 960, rotation_deg=-90)
    payload = probe(output)
    stream = next(item for item in payload["streams"] if item["codec_type"] == "video")
    assert int(stream["width"]) == 240
    assert int(stream["height"]) == 320
    untouched = tmp_path / "wide-out.mp4"
    normalize(source, untouched, 960, rotation_deg=0)
    plain = next(item for item in probe(untouched)["streams"] if item["codec_type"] == "video")
    assert int(plain["width"]) == 320
    assert int(plain["height"]) == 240


def test_mp4v_intermediate_becomes_h264(tmp_path: Path):
    import cv2

    raw = tmp_path / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    frame = __import__("numpy").zeros((48, 64, 3), dtype="uint8")
    for _ in range(5):
        writer.write(frame)
    writer.release()
    dest = tmp_path / "play.mp4"
    transcode_h264(raw, dest)
    from primed.video_io import probe

    stream = next(item for item in probe(dest)["streams"] if item["codec_type"] == "video")
    assert stream["codec_name"] == "h264"


def test_timestamps_are_monotonic():
    assert monotonic_milliseconds([0.0, 0.001, 0.001, 0.01]) == [0, 1, 2, 10]


def test_blank_clip_does_not_invent_measurements(tmp_path: Path):
    path = tmp_path / "empty.mp4"
    _color_clip(path, duration=0.45, size="160x120")
    repository = LocalRepository(tmp_path / "session")
    workspace = new_workspace()
    clip = ingest_upload(
        repository,
        workspace,
        path.read_bytes(),
        "empty.mp4",
        {"profile_id": "squat", "variant": "air", "view": "side", "facing": "right", "context": {}},
        limits=Limits(max_width=160, sample_fps=8, max_frames=20),
    )
    analysis = analyze_clip(repository, workspace, clip["id"], limits=Limits(max_width=160, sample_fps=8, max_frames=20))
    assert analysis["tracking_fraction"] == 0
    assert analysis["reps"] == []
    assert any("No pose" in warning for warning in analysis["warnings"])
    assert (repository.clip_dir(clip["id"]) / "playback.mp4").is_file()
    # A second call with the same parameters must not be required to detect a pose either.
    again = analyze_clip(repository, workspace, clip["id"], limits=Limits(max_width=160, sample_fps=8, max_frames=20))
    assert again["id"] == analysis["id"]


def test_overlay_gap_is_drawn(tmp_path: Path):
    import cv2
    import numpy as np

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    drawn = draw_pose(frame, None)
    assert drawn[70, 10].sum() > 0


def test_cloud_flag_does_not_silently_share_storage(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRIMED_ENABLE_CLOUD", "1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(CloudNotConfigured):
        open_repository(tmp_path)
