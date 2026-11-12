"""Baseline versions, drift, and activity scope."""

from datetime import datetime, timedelta, timezone

from primed.baseline import (
    active_reference,
    approve_reference,
    build_trend,
    local_day,
    select_reference,
    set_baseline,
    standardised_deviation,
)
from primed.measure import apply_rep_edit, derive_analysis
from primed.workspace import export_workspace, import_workspace, new_workspace


def _squat(bottom: float, when: datetime, view: str = "side", variant: str = "air", quality_hole: bool = False):
    times = [i * 0.1 for i in range(30)]
    knee = []
    for moment in times:
        if moment < 0.4:
            knee.append(170)
        elif moment < 1.0:
            knee.append(170 - (170 - bottom) * ((moment - 0.4) / 0.6))
        elif moment < 1.2:
            knee.append(bottom)
        elif moment < 1.8:
            knee.append(bottom + (170 - bottom) * ((moment - 1.2) / 0.6))
        else:
            knee.append(170)
    trunk = [None if quality_hole else 12.0] * len(times)
    return derive_analysis(
        profile_id="squat",
        times=times,
        series={"knee_left": knee, "hip_left": [value - 5 for value in knee], "trunk_tilt": trunk},
        meta={
            "view": view,
            "facing": "right",
            "variant": variant,
            "recorded_at": when.replace(microsecond=0).isoformat(),
            "athlete": "Athlete",
            "synthetic": True,
            "source_kind": "unit",
            "context": {"load_kg": 60},
        },
    )


def _park(workspace, analysis, set_label: str | None):
    workspace["analyses"][analysis["id"]] = analysis
    if set_label and analysis["reps"]:
        rep = analysis["reps"][0]
        apply_rep_edit(analysis, rep["id"], start_s=rep["start_s"], end_s=rep["end_s"], set_label=set_label)


def test_single_rep_reference_has_raw_change_and_no_sigma():
    workspace = new_workspace(timezone_name="UTC")
    analysis = _squat(80, datetime(2024, 6, 1, tzinfo=timezone.utc))
    _park(workspace, analysis, "Set 1")
    other = _squat(90, datetime(2024, 6, 1, 1, tzinfo=timezone.utc))
    _park(workspace, other, "Set 1")
    reference = set_baseline(workspace, [analysis["reps"][0]["id"]])
    assert reference["version"] == 1
    knee = reference["scalars"]["bottom_knee_angle"]
    assert knee["n"] == 1
    assert knee["sigma_ok"] is False
    assert "variance" in knee["sigma_reason"].lower() or "single" in knee["sigma_reason"].lower()
    trend = build_trend(workspace, reference, mode="reps", standardised=False, timezone_name="UTC")
    series = next(item for item in trend["series"] if item["metric_id"] == "bottom_knee_angle")
    values = [point["y"] for point in series["points"]]
    assert values[0] == 0
    assert abs(values[1] - 10) < 0.2
    sigma = build_trend(workspace, reference, mode="reps", standardised=True, timezone_name="UTC")
    sigma_series = next(item for item in sigma["series"] if item["metric_id"] == "bottom_knee_angle")
    assert sigma_series["unavailable_reason"]
    assert all(point["y"] is None for point in sigma_series["points"])


def test_five_reps_enable_sigma_and_zero_spread_suppresses_it():
    workspace = new_workspace()
    ids = []
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for index, bottom in enumerate([80, 82, 88, 95, 100]):
        analysis = _squat(bottom, start + timedelta(minutes=index))
        _park(workspace, analysis, "Set 1")
        ids.append(analysis["reps"][0]["id"])
    reference = set_baseline(workspace, ids)
    scalar = reference["scalars"]["bottom_knee_angle"]
    assert scalar["n"] == 5
    assert scalar["sigma_ok"] is True
    assert abs(scalar["value"] - (80 + 82 + 88 + 95 + 100) / 5) < 0.05
    observed = scalar["value"] + scalar["std"]
    assert abs(standardised_deviation(observed, scalar["value"], scalar["std"], True) - 1) < 1e-6
    flat = new_workspace()
    flat_ids = []
    for index in range(5):
        analysis = _squat(80, start + timedelta(minutes=index))
        _park(flat, analysis, "Set 1")
        flat_ids.append(analysis["reps"][0]["id"])
    flat_ref = set_baseline(flat, flat_ids)
    assert flat_ref["scalars"]["bottom_knee_angle"]["sigma_ok"] is False
    assert "identical" in flat_ref["scalars"]["bottom_knee_angle"]["sigma_reason"]


def test_versions_stay_immutable_and_selection_recalculates():
    workspace = new_workspace()
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    analyses = []
    for index, bottom in enumerate([70, 80, 90, 100, 110, 120]):
        analysis = _squat(bottom, start + timedelta(minutes=index))
        _park(workspace, analysis, f"Set {index // 2 + 1}")
        analyses.append(analysis)
    first = set_baseline(workspace, [analyses[0]["reps"][0]["id"]])
    second = set_baseline(workspace, [item["reps"][0]["id"] for item in analyses[:5]])
    assert first["version"] == 1
    assert second["version"] == 2
    assert first["scalars"]["bottom_knee_angle"]["value"] != second["scalars"]["bottom_knee_angle"]["value"]
    assert active_reference(workspace, "squat", "air", "side")["id"] == second["id"]
    select_reference(workspace, first["id"])
    assert active_reference(workspace, "squat", "air", "side")["id"] == first["id"]
    # Stored numbers on version 2 did not change when it stopped being active.
    assert get_value(workspace, second["id"]) == second["scalars"]["bottom_knee_angle"]["value"]
    trend = build_trend(workspace, first, mode="reps", standardised=False)
    series = next(item for item in trend["series"] if item["metric_id"] == "bottom_knee_angle")
    assert series["points"][0]["y"] == 0
    later = build_trend(workspace, second, mode="sets", standardised=False)
    set_series = next(item for item in later["series"] if item["metric_id"] == "bottom_knee_angle")
    assert any(point["n"] >= 1 and point["y"] is not None for point in set_series["points"])


def get_value(workspace, reference_id):
    for ref in workspace["references"]:
        if ref["id"] == reference_id:
            return ref["scalars"]["bottom_knee_angle"]["value"]
    raise AssertionError(reference_id)


def test_sets_use_median_and_days_follow_the_workspace_timezone():
    workspace = new_workspace(timezone_name="America/Los_Angeles")
    # 06:30 UTC is 23:30 the previous evening in Los Angeles (PDT, UTC−7). 07:30 UTC is 00:30.
    early = _squat(80, datetime(2024, 6, 2, 6, 30, tzinfo=timezone.utc))
    late = _squat(100, datetime(2024, 6, 2, 7, 30, tzinfo=timezone.utc))
    _park(workspace, early, "Set 1")
    _park(workspace, late, "Set 1")
    assert local_day(early["recorded_at"], "America/Los_Angeles") == "2024-06-01"
    assert local_day(late["recorded_at"], "America/Los_Angeles") == "2024-06-02"
    extras = []
    start = datetime(2024, 6, 3, tzinfo=timezone.utc)
    for index, bottom in enumerate([82, 84, 86]):
        analysis = _squat(bottom, start + timedelta(days=index))
        _park(workspace, analysis, "Set 2")
        extras.append(analysis)
    reference = set_baseline(
        workspace,
        [early["reps"][0]["id"]] + [item["reps"][0]["id"] for item in extras] + [late["reps"][0]["id"]],
    )
    # Five values: 80, 82, 84, 86, 100. Not identical.
    assert reference["scalars"]["bottom_knee_angle"]["n"] == 5
    trend = build_trend(workspace, reference, mode="sets", standardised=False, timezone_name="America/Los_Angeles")
    series = next(item for item in trend["series"] if item["metric_id"] == "bottom_knee_angle")
    by_label = {point["x"]: point for point in series["points"]}
    # Set 1 holds 80 and 100. Median signed change is the median of the two signed values.
    assert by_label["Set 1"]["n"] == 2
    days = build_trend(workspace, reference, mode="days", standardised=False, timezone_name="America/Los_Angeles")
    day_series = next(item for item in days["series"] if item["metric_id"] == "bottom_knee_angle")
    labels = [point["x"] for point in day_series["points"]]
    assert "2024-06-01" in labels and "2024-06-02" in labels


def test_other_movement_cannot_reuse_the_squat_reference():
    workspace = new_workspace()
    squat = _squat(85, datetime(2024, 6, 1, tzinfo=timezone.utc))
    _park(workspace, squat, "Set 1")
    times = [i * 0.1 for i in range(40)]
    hip = [100 if moment < 0.5 or moment > 2.0 else 170 for moment in times]
    # Build a rise that the deadlift detector can see.
    hip = []
    knee = []
    for moment in times:
        if moment < 0.4:
            hip.append(100)
            knee.append(130)
        elif moment < 1.5:
            hip.append(100 + 70 * (moment - 0.4) / 1.1)
            knee.append(140)
        elif moment < 2.4:
            hip.append(170 - 70 * (moment - 1.5) / 0.9)
            knee.append(140)
        else:
            hip.append(100)
            knee.append(130)
    deadlift = derive_analysis(
        profile_id="deadlift",
        times=times,
        series={"hip_left": hip, "knee_left": knee, "trunk_tilt": [20] * len(times)},
        meta={"view": "side", "facing": "right", "variant": "conventional", "recorded_at": "2024-06-01T00:00:00+00:00", "synthetic": True},
    )
    workspace["analyses"][deadlift["id"]] = deadlift
    reference = set_baseline(workspace, [squat["reps"][0]["id"]])
    trend = build_trend(workspace, reference, mode="reps", standardised=False)
    reasons = {item["reason"] for item in trend["excluded"]}
    assert "Different movement" in reasons
    assert active_reference(workspace, "deadlift", "conventional", "side") is None
    try:
        set_baseline(workspace, [squat["reps"][0]["id"], deadlift["reps"][0]["id"]])
        raise AssertionError("mixed movements should be rejected")
    except ValueError as exc:
        assert "one movement" in str(exc)


def test_incompatible_view_is_excluded_and_missing_metric_is_a_gap():
    workspace = new_workspace()
    good = []
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for index, bottom in enumerate([80, 85, 90, 95, 100]):
        analysis = _squat(bottom, start + timedelta(minutes=index), quality_hole=(index == 4))
        _park(workspace, analysis, "Set 1")
        good.append(analysis)
    front = _squat(90, start + timedelta(hours=2), view="front")
    workspace["analyses"][front["id"]] = front
    reference = set_baseline(workspace, [item["reps"][0]["id"] for item in good[:4]])
    # The fifth good rep has no trunk samples. Knee still works. Trunk reference n may be 4.
    trend = build_trend(workspace, reference, mode="reps", standardised=False)
    reasons = [item["reason"] for item in trend["excluded"]]
    assert "Different camera view" in reasons
    trunk = next(item for item in trend["series"] if item["metric_id"] == "trunk_tilt")
    assert any(point["y"] is None for point in trunk["points"])
    knee = next(item for item in trend["series"] if item["metric_id"] == "bottom_knee_angle")
    assert any(point["y"] is not None for point in knee["points"])


def test_export_roundtrip_drops_media_and_keeps_measurements():
    workspace = new_workspace()
    analysis = _squat(88, datetime(2024, 6, 1, tzinfo=timezone.utc))
    _park(workspace, analysis, "Set 1")
    reference = set_baseline(workspace, [analysis["reps"][0]["id"]])
    workspace["clips"]["clip_abc123"] = {
        "id": "clip_abc123",
        "filename": "../../secret.mp4",
        "sha256": "a" * 64,
        "path": "/tmp/secret.mp4",
        "media_present": True,
        "profile_id": "squat",
    }
    analysis["clip_id"] = "clip_abc123"
    exported = export_workspace(workspace)
    assert exported["export_version"] == 1
    assert exported["media_included"] is False
    assert exported["clips"]["clip_abc123"]["media_present"] is False
    assert "path" not in exported["clips"]["clip_abc123"]
    assert exported["clips"]["clip_abc123"]["filename"] == "secret.mp4"
    assert exported["clips"]["clip_abc123"]["sha256"] == "a" * 64
    imported = import_workspace(exported)
    restored = imported["analyses"][analysis["id"]]["reps"][0]["metrics"]["bottom_knee_angle"]["value"]
    assert restored == analysis["reps"][0]["metrics"]["bottom_knee_angle"]["value"]
    assert imported["references"][0]["id"] == reference["id"]
    assert imported["clips"]["clip_abc123"]["media_present"] is False


def test_import_rejects_a_bad_version_and_a_bad_unit():
    workspace = new_workspace()
    analysis = _squat(88, datetime(2024, 6, 1, tzinfo=timezone.utc))
    _park(workspace, analysis, None)
    set_baseline(workspace, [analysis["reps"][0]["id"]])
    exported = export_workspace(workspace)
    exported["export_version"] = 99
    try:
        import_workspace(exported)
        raise AssertionError("expected a version error")
    except ValueError as exc:
        assert "version" in str(exc).lower()
    exported["export_version"] = 1
    rep = next(iter(exported["analyses"].values()))["reps"][0]
    rep["metrics"]["bottom_knee_angle"]["unit"] = "m"
    try:
        import_workspace(exported)
        raise AssertionError("expected a unit error")
    except ValueError as exc:
        assert "bottom_knee_angle" in str(exc)


def test_coach_approval_keeps_the_same_reference_id():
    workspace = new_workspace()
    analysis = _squat(88, datetime(2024, 6, 1, tzinfo=timezone.utc))
    _park(workspace, analysis, None)
    reference = set_baseline(workspace, [analysis["reps"][0]["id"]], provenance="self-selected")
    approved = approve_reference(workspace, reference["id"], "coach-approved")
    assert approved["id"] == reference["id"]
    assert approved["provenance"] == "coach-approved"
    assert approved["approvals"][-1]["action"] == "coach-approved"
