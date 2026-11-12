"""Download the official Pose Landmarker lite model.

The URL is the versioned MediaPipe asset, not the floating "latest" alias.
The file is checked against a recorded sha256 before it is kept.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primed.config import POSE_MODEL_SHA256, POSE_MODEL_URL, model_path


def main() -> None:
    destination = model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == POSE_MODEL_SHA256:
        print(f"Pose model already present at {destination}")
        return
    temporary = destination.with_suffix(".task.partial")
    print(f"Downloading {POSE_MODEL_URL}")
    urllib.request.urlretrieve(POSE_MODEL_URL, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != POSE_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Model hash mismatch: {digest}")
    temporary.replace(destination)
    print(f"Saved {destination} ({digest})")


if __name__ == "__main__":
    main()
