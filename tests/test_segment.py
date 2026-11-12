"""Rep detection and manual boundary tests."""

from primed.measure import add_manual_rep, apply_rep_edit, derive_analysis, resegment
from primed.series import interpolate_short_gaps


def _knee_dip(bottom: float, fps: float = 10.0) -> tuple[list[float], list[float | None]]:
    # Stand, descend, bottom, ascend, stand. About 1.6 s of motion plus rest.
    pieces = [
        (0.5, 170, 170),
        (0.6, 170, bottom),
        (0.2, bottom, bottom),
        (0.6, bottom, 170),
        (0.4, 170, 170),
    ]
    times: list[float] = []
    values: list[float | None] = []
    cursor = 0.0
    for duration, start, end in pieces:
        steps = max(2, int(round(duration * fps)))
        for step in range(steps):
            if times and step == 0:
                continue
            fraction = step / (steps - 1)
            times.append(round(cursor + duration * fraction, 4))
            values.append(start + (end - start) * fraction)
        cursor += duration
    return times, values


def test_squat_finds_two_reps_and_keeps_raw_timing():
    first_t, first_v = _knee_dip(80)
    second_t, second_v = _knee_dip(70)
    offset = first_t[-1] + 0.2
    times = first_t + [round(offset + value, 4) for value in second_t]
    knee = first_v + second_v
    hip = [value + 5 if value is not None else None for value in knee]
    trunk = [15.0] * len(times)
    analysis = derive_analysis(
        profile_id="squat",
        times=times,
        series={"knee_left": knee, "hip_left": hip, "trunk_tilt": trunk, "knee_right": [None] * len(times)},
        meta={"view": "side", "facing": "right", "variant": "air", "synthetic": True, "source_kind": "unit"},
    )
    reps = [rep for rep in analysis["reps"] if rep["quality"] == "ok"]
    assert len(reps) == 2
    first = reps[0]
    descent = first["metrics"]["descent_s"]
    assert descent["valid"] is True
    assert descent["unit"] == "s"
    # Timing is the phase's own timestamps, not a normalised 0–1 width.
    phase = next(item for item in first["phases"] if item["id"] == "descent")
    assert abs(descent["value"] - (phase["end_s"] - phase["start_s"])) < 1e-6
    assert first["metrics"]["bottom_knee_angle"]["valid"] is True
    assert first["metrics"]["bottom_knee_angle"]["value"] < 90
    assert "2D" in first["metrics"]["bottom_knee_angle"]["reason"] or first["metrics"]["bottom_knee_angle"]["unit"] == "deg"


def test_manual_edit_survives_resegmentation():
    times, knee = _knee_dip(82)
    analysis = derive_analysis(
        profile_id="squat",
        times=times,
        series={"knee_left": knee, "hip_left": knee, "trunk_tilt": [10] * len(times)},
        meta={"view": "side", "facing": "right", "variant": "air"},
    )
    assert analysis["reps"]
    rep = analysis["reps"][0]
    edited_start = round(rep["start_s"] + 0.1, 4)
    apply_rep_edit(analysis, rep["id"], start_s=edited_start, end_s=rep["end_s"], set_label="Set 1")
    resegment(analysis)
    kept = next(item for item in analysis["reps"] if item["id"] == rep["id"])
    assert kept["edited"] is True
    assert kept["start_s"] == edited_start
    assert kept["set_label"] == "Set 1"


def test_manual_movement_has_no_automatic_rep():
    times = [i * 0.1 for i in range(20)]
    analysis = derive_analysis(
        profile_id="golf_swing",
        times=times,
        series={"elbow_left": [150.0] * len(times), "trunk_tilt": [5.0] * len(times)},
        meta={"view": "down_the_line", "facing": "right", "variant": "full", "handedness": "right"},
    )
    assert analysis["reps"] == []
    rep = add_manual_rep(analysis)
    assert rep["source"] == "manual"
    assert rep["metrics"]["elbow_min"]["valid"] is True
    assert rep["metrics"]["backswing_s"]["valid"] is False


def test_confirming_events_makes_duration_usable():
    times = [i * 0.1 for i in range(30)]
    analysis = derive_analysis(
        profile_id="golf_swing",
        times=times,
        series={"elbow_left": [140.0] * len(times)},
        meta={"view": "down_the_line", "facing": "right", "variant": "full", "handedness": "right"},
    )
    rep = add_manual_rep(analysis, 0.2, 2.4)
    apply_rep_edit(
        analysis,
        rep["id"],
        start_s=0.2,
        end_s=2.4,
        set_label=None,
        event_times={"address": 0.3, "top": 1.1, "impact": 1.4, "follow_through": 2.2},
        confirm_ids={"address", "top", "impact", "follow_through"},
    )
    updated = analysis["reps"][0]
    assert updated["metrics"]["backswing_s"]["valid"] is True
    assert abs(updated["metrics"]["backswing_s"]["value"] - 0.8) < 1e-6


def test_deadlift_marks_knee_pass_as_candidate():
    times = [i * 0.1 for i in range(40)]
    hip = []
    knee = []
    for moment in times:
        if moment < 0.4:
            hip.append(100)
            knee.append(120)
        elif moment < 1.4:
            fraction = (moment - 0.4) / 1.0
            hip.append(100 + 70 * fraction)
            knee.append(120 + 55 * fraction)
        elif moment < 2.2:
            fraction = (moment - 1.4) / 0.8
            hip.append(170 - 70 * fraction)
            knee.append(175 - 50 * fraction)
        else:
            hip.append(100)
            knee.append(120)
    analysis = derive_analysis(
        profile_id="deadlift",
        times=times,
        series={"hip_left": hip, "knee_left": knee, "trunk_tilt": [20] * len(times)},
        meta={"view": "side", "facing": "right", "variant": "conventional"},
    )
    assert len(analysis["reps"]) == 1
    knee_pass = next(event for event in analysis["reps"][0]["events"] if event["id"] == "knee_pass")
    assert knee_pass["status"] == "candidate"
    assert knee_pass["t"] is not None
    assert "bar" in knee_pass["note"].lower()


def test_cricket_release_stays_a_candidate_while_elbow_is_measured():
    times = [i * 0.05 for i in range(40)]
    speed = [5.0] * len(times)
    speed[20] = 200.0
    analysis = derive_analysis(
        profile_id="cricket_bowling",
        times=times,
        series={
            "wrist_speed_right": speed,
            "elbow_right": [140.0] * len(times),
            "knee_left": [150.0] * len(times),
            "trunk_tilt": [12.0] * len(times),
        },
        meta={"view": "side", "facing": "right", "variant": "pace", "handedness": "right"},
    )
    assert len(analysis["reps"]) == 1
    rep = analysis["reps"][0]
    assert rep["quality"] == "uncertain"
    release = next(event for event in rep["events"] if event["id"] == "release")
    assert release["status"] == "candidate"
    assert rep["metrics"]["bowling_elbow"]["valid"] is True
    assert rep["metrics"]["delivery_s"]["valid"] is False


def test_long_gaps_stay_empty_and_raw_series_are_not_filled():
    times = [0, 0.1, 0.2, 1.0, 1.1]
    raw = [1.0, None, 3.0, None, 5.0]
    plotted, filled = interpolate_short_gaps(raw, times, max_gap_s=0.2)
    assert plotted[1] == 2.0
    assert filled[1] is True
    assert plotted[3] is None
    assert raw[1] is None


def test_front_view_squat_metrics_say_they_need_a_side_view():
    times, knee = _knee_dip(90)
    analysis = derive_analysis(
        profile_id="squat",
        times=times,
        series={"knee_left": knee, "hip_left": knee, "trunk_tilt": [0] * len(times)},
        meta={"view": "front", "facing": "", "variant": "air"},
    )
    if analysis["reps"]:
        metric = analysis["reps"][0]["metrics"]["bottom_knee_angle"]
        assert metric["valid"] is False
        assert "side" in metric["reason"]
