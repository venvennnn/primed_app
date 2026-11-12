"""A labelled synthetic workspace for chart checks.

These records are produced by the same measurement code as an upload, from
constructed angle series. They are not a filmed session and the workspace is
marked synthetic. There is no clip media.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from primed.baseline import set_baseline
from primed.measure import apply_rep_edit, derive_analysis
from primed.workspace import new_workspace


def _dip_signal(bottom: float) -> tuple[list[float], list[float]]:
    times = [round(index * 0.1, 4) for index in range(30)]
    knee = []
    for moment in times:
        if moment < 0.4:
            knee.append(170.0)
        elif moment < 1.0:
            knee.append(170.0 - (170.0 - bottom) * ((moment - 0.4) / 0.6))
        elif moment < 1.2:
            knee.append(float(bottom))
        elif moment < 1.8:
            knee.append(bottom + (170.0 - bottom) * ((moment - 1.2) / 0.6))
        else:
            knee.append(170.0)
    return times, knee


def _park(workspace: dict, analysis: dict, set_label: str | None, filename: str) -> None:
    analysis["filename"] = filename
    workspace["analyses"][analysis["id"]] = analysis
    if set_label and analysis["reps"]:
        rep = analysis["reps"][0]
        apply_rep_edit(analysis, rep["id"], start_s=rep["start_s"], end_s=rep["end_s"], set_label=set_label)


def build_synthetic_workspace() -> dict:
    start = datetime(2024, 6, 3, 16, 0, tzinfo=timezone.utc)
    workspace = new_workspace(name="Synthetic fixture", athlete="Fixture athlete", timezone_name="America/Los_Angeles")
    workspace["synthetic"] = True
    bottoms = [80, 82, 88, 95, 100, 110, 120, 125]
    analyses = []
    for index, bottom in enumerate(bottoms):
        times, knee = _dip_signal(bottom)
        trunk = [None] * len(times) if index == len(bottoms) - 1 else [18.0] * len(times)
        hip_y = [250.0 if value < 140 else 200.0 for value in knee]
        analysis = derive_analysis(
            profile_id="squat",
            times=times,
            series={
                "knee_left": knee,
                "hip_left": [value - 8 for value in knee],
                "trunk_tilt": trunk,
                "hip_y": hip_y,
                "torso_len": [100.0] * len(times),
            },
            meta={
                "view": "side",
                "facing": "right",
                "variant": "air",
                "handedness": "",
                "recorded_at": (start + timedelta(days=index // 4, minutes=index)).isoformat(),
                "athlete": "Fixture athlete",
                "synthetic": True,
                "source_kind": "synthetic_fixture",
                "context": {"load_kg": 60},
                "filename": f"synthetic-squat-{index + 1}.json",
                "width": 640,
                "height": 480,
                "timing_mode": "synthetic",
            },
        )
        day = 1 if index < 4 else 2
        _park(workspace, analysis, f"Set {index // 2 + 1}", f"Synthetic squat day {day} rep {index + 1}")
        analyses.append(analysis)
    front_times, front_knee = _dip_signal(90)
    front = derive_analysis(
        profile_id="squat",
        times=front_times,
        series={"knee_left": front_knee, "hip_left": front_knee, "trunk_tilt": [0.0] * len(front_times)},
        meta={
            "view": "front",
            "facing": "",
            "variant": "air",
            "recorded_at": (start + timedelta(days=2)).isoformat(),
            "athlete": "Fixture athlete",
            "synthetic": True,
            "source_kind": "synthetic_fixture",
            "context": {"load_kg": 60},
            "filename": "synthetic-squat-front.json",
        },
    )
    workspace["analyses"][front["id"]] = front
    times = [round(index * 0.1, 4) for index in range(40)]
    hip = []
    knee = []
    for moment in times:
        if moment < 0.4 or moment > 2.3:
            hip.append(100.0)
            knee.append(125.0)
        elif moment < 1.4:
            hip.append(100.0 + 70.0 * (moment - 0.4))
            knee.append(125.0 + 50.0 * (moment - 0.4))
        else:
            hip.append(170.0 - 70.0 * (moment - 1.4) / 0.9)
            knee.append(175.0 - 50.0 * (moment - 1.4) / 0.9)
    deadlift = derive_analysis(
        profile_id="deadlift",
        times=times,
        series={"hip_left": hip, "knee_left": knee, "trunk_tilt": [25.0] * len(times)},
        meta={
            "view": "side",
            "facing": "right",
            "variant": "conventional",
            "recorded_at": (start + timedelta(days=1)).isoformat(),
            "athlete": "Fixture athlete",
            "synthetic": True,
            "source_kind": "synthetic_fixture",
            "context": {"load_kg": 100},
            "filename": "synthetic-deadlift.json",
        },
    )
    if deadlift["reps"]:
        rep = deadlift["reps"][0]
        apply_rep_edit(deadlift, rep["id"], start_s=rep["start_s"], end_s=rep["end_s"], set_label="Set 1")
    workspace["analyses"][deadlift["id"]] = deadlift
    set_baseline(workspace, [analyses[0]["reps"][0]["id"]], provenance="self-selected")
    set_baseline(workspace, [item["reps"][0]["id"] for item in analyses[:5]], provenance="self-selected")
    return workspace
