"""Measure one clip from the terminal.

Example:
    python -m primed.cli clip.mp4 --profile squat --view side --facing right
"""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from pathlib import Path

from primed.analyze import analyze_clip, ingest_upload
from primed.repository import LocalRepository
from primed.workspace import new_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a clip with Primed.")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--profile", default="squat")
    parser.add_argument("--variant", default="")
    parser.add_argument("--view", default="side")
    parser.add_argument("--facing", default="right")
    parser.add_argument("--handedness", default="")
    parser.add_argument("--load", type=float, default=None)
    args = parser.parse_args(argv)
    from primed.profiles import get_profile

    profile = get_profile(args.profile)
    variant = args.variant or profile.default_variant
    data = args.clip.read_bytes()
    with tempfile.TemporaryDirectory(prefix="primed-cli-") as temporary:
        repository = LocalRepository(Path(temporary) / uuid.uuid4().hex)
        workspace = new_workspace()
        context = {}
        if args.load is not None:
            context["load_kg"] = args.load
        clip = ingest_upload(
            repository,
            workspace,
            data,
            args.clip.name,
            {
                "profile_id": profile.id,
                "variant": variant,
                "view": args.view,
                "facing": args.facing,
                "handedness": args.handedness,
                "context": context,
            },
        )
        analysis = analyze_clip(repository, workspace, clip["id"])
    summary = {
        "filename": clip["filename"],
        "profile": profile.name,
        "timing_mode": analysis.get("timing_mode"),
        "tracking_fraction": analysis.get("tracking_fraction"),
        "warnings": analysis.get("warnings"),
        "reps": [
            {
                "id": rep["id"],
                "start_s": rep["start_s"],
                "end_s": rep["end_s"],
                "quality": rep["quality"],
                "metrics": {
                    key: {"value": value.get("value"), "unit": value.get("unit"), "valid": value.get("valid")}
                    for key, value in rep.get("metrics", {}).items()
                },
            }
            for rep in analysis.get("reps", [])
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
