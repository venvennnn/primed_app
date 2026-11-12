"""Derive measurements from landmarks or from already computed series.

Observations and derived rep metrics are stored separately. Metrics read raw
samples. Plot series are a separate field and may bridge gaps shorter than
the configured limit.
"""

from __future__ import annotations

import math
from typing import Any

from primed.config import SHORT_GAP_S, VISIBILITY_MIN
from primed.geometry import hip_mid_y_px, joint_angle_px, torso_length_px, trunk_tilt_deg
from primed.ids import new_id
from primed.profiles import MovementProfile, get_profile
from primed.segment import ProposedRep, apply_knee_pass, delivery_stride_time, segment_signal
from primed.series import finite, interpolate_short_gaps, nearest_value

# BlazePose indices.
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_ELBOW, _RIGHT_ELBOW = 13, 14
_LEFT_WRIST, _RIGHT_WRIST = 15, 16
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_KNEE, _RIGHT_KNEE = 25, 26
_LEFT_ANKLE, _RIGHT_ANKLE = 27, 28


def _visibility(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        visibility = getattr(landmark, "presence", 0.0)
    if visibility is None:
        return 0.0
    return float(visibility)


def landmarks_from_result(result: Any) -> list[list[float]] | None:
    poses = getattr(result, "pose_landmarks", None) or []
    if not poses:
        return None
    pose = poses[0]
    converted = []
    for landmark in pose:
        converted.append(
            [
                round(float(landmark.x), 5),
                round(float(landmark.y), 5),
                round(_visibility(landmark), 4),
            ]
        )
    if len(converted) < 33:
        return None
    return converted


def series_from_landmarks(
    frames: list[dict],
    width: int,
    height: int,
    facing: str,
    min_visibility: float = VISIBILITY_MIN,
) -> dict[str, list[float | None]]:
    names = [
        "elbow_left",
        "elbow_right",
        "knee_left",
        "knee_right",
        "hip_left",
        "hip_right",
        "trunk_tilt",
        "wrist_speed_left",
        "wrist_speed_right",
        "hip_y",
        "torso_len",
    ]
    series: dict[str, list[float | None]] = {name: [] for name in names}
    previous_wrist: dict[str, tuple[float, float, float] | None] = {"left": None, "right": None}
    for frame in frames:
        landmarks = frame.get("lm")
        moment = float(frame["t"])
        series["elbow_left"].append(
            joint_angle_px(landmarks, width, height, _LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST, min_visibility)
        )
        series["elbow_right"].append(
            joint_angle_px(landmarks, width, height, _RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST, min_visibility)
        )
        series["knee_left"].append(
            joint_angle_px(landmarks, width, height, _LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE, min_visibility)
        )
        series["knee_right"].append(
            joint_angle_px(landmarks, width, height, _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE, min_visibility)
        )
        series["hip_left"].append(
            joint_angle_px(landmarks, width, height, _LEFT_SHOULDER, _LEFT_HIP, _LEFT_KNEE, min_visibility)
        )
        series["hip_right"].append(
            joint_angle_px(landmarks, width, height, _RIGHT_SHOULDER, _RIGHT_HIP, _RIGHT_KNEE, min_visibility)
        )
        tilt = trunk_tilt_deg(landmarks, width, height, facing, min_visibility) if facing in ("left", "right") else None
        series["trunk_tilt"].append(tilt)
        series["hip_y"].append(hip_mid_y_px(landmarks, width, height, min_visibility))
        series["torso_len"].append(torso_length_px(landmarks, width, height, min_visibility))
        for side, index in (("left", _LEFT_WRIST), ("right", _RIGHT_WRIST)):
            key = f"wrist_speed_{side}"
            if landmarks is None or landmarks[index][2] < min_visibility:
                series[key].append(None)
                previous_wrist[side] = None
                continue
            x = landmarks[index][0] * width
            y = landmarks[index][1] * height
            previous = previous_wrist[side]
            previous_wrist[side] = (moment, x, y)
            if previous is None:
                series[key].append(None)
                continue
            dt = moment - previous[0]
            if dt <= 1e-6:
                series[key].append(None)
                continue
            speed = math.hypot(x - previous[1], y - previous[2]) / dt
            series[key].append(speed)
    return series


def _coverage(values: list[float | None]) -> tuple[int, float]:
    usable = [value for value in values if value is not None]
    if not values:
        return 0, 0.0
    return len(usable), (max(usable) - min(usable)) if usable else 0.0


def choose_side(
    left: list[float | None] | None,
    right: list[float | None] | None,
    preference: str | None = None,
    *,
    strict: bool = False,
) -> str | None:
    """Pick a side. A strict preference is kept even when that side is hidden."""
    if preference in ("left", "right"):
        preferred = left if preference == "left" else right
        count, _range = _coverage(preferred or [])
        if strict or count >= 2:
            return preference
    left = left or []
    right = right or []
    left_n, left_range = _coverage(left)
    right_n, right_range = _coverage(right)
    if left_n == 0 and right_n == 0:
        return None
    if left_n >= 3 and (right_n < 3 or left_range >= right_range):
        return "left"
    if right_n >= 3:
        return "right"
    return "left" if left_n >= right_n else "right"


def _alias_series(series: dict[str, list[float | None]], sides: dict[str, str | None]) -> dict[str, list[float | None]]:
    """Add chosen-side aliases without erasing a series the caller already provided."""
    aliased = dict(series)
    count = _length(series)
    for joint in ("elbow", "knee", "hip"):
        side = sides.get(joint)
        side_key = f"{joint}_{side}" if side in ("left", "right") else ""
        if side_key and side_key in series:
            aliased[joint] = list(series[side_key])
        elif joint not in aliased:
            aliased[joint] = [None] * count
    hand = sides.get("wrist")
    hand_key = f"wrist_speed_{hand}" if hand in ("left", "right") else ""
    if hand_key and hand_key in series:
        aliased["wrist_speed"] = list(series[hand_key])
    elif "wrist_speed" not in aliased:
        aliased["wrist_speed"] = [None] * count
    return aliased


def _length(series: dict[str, list[float | None]]) -> int:
    if not series:
        return 0
    return max(len(values) for values in series.values())


def _window(times: list[float], values: list[float | None], start_s: float, end_s: float) -> list[float]:
    return [
        value
        for moment, value in zip(times, values)
        if start_s - 1e-9 <= moment <= end_s + 1e-9 and value is not None
    ]


def _metric_payload(
    value: float | None,
    unit: str,
    status: str,
    reason: str = "",
) -> dict:
    valid = value is not None and status in ("measured", "confirmed")
    return {
        "value": None if value is None else round(float(value), 4),
        "unit": unit,
        "valid": valid,
        "status": status if value is not None or status == "unavailable" else "unavailable",
        "reason": reason,
    }


def _event_lookup(rep: dict) -> dict[str, dict]:
    return {event["id"]: event for event in rep["events"]}


def _phase_lookup(rep: dict) -> dict[str, dict]:
    return {phase["id"]: phase for phase in rep["phases"]}


def compute_metrics(
    profile: MovementProfile,
    rep: dict,
    times: list[float],
    series: dict[str, list[float | None]],
    view: str,
    sample_period: float,
) -> dict[str, dict]:
    events = _event_lookup(rep)
    phases = _phase_lookup(rep)
    metrics: dict[str, dict] = {}
    for spec in profile.metrics:
        if view not in spec.views:
            metrics[spec.id] = _metric_payload(None, spec.unit, "unavailable", f"Needs a {' or '.join(spec.views)} view.")
            continue
        status = "measured"
        reason = spec.note
        value: float | None = None
        if spec.reducer in ("min", "max", "median"):
            source = series.get(spec.series or "", [])
            samples = _window(times, source, rep["start_s"], rep["end_s"])
            if len(samples) < 2:
                status = "unavailable"
                reason = "Not enough visible frames for this measurement."
            elif spec.reducer == "min":
                value = min(samples)
            elif spec.reducer == "max":
                value = max(samples)
            else:
                ordered = sorted(samples)
                value = ordered[len(ordered) // 2]
        elif spec.reducer == "at_event":
            event = events.get(spec.event_id or "")
            source = series.get(spec.series or "", [])
            if event is None or event.get("t") is None:
                status = "unavailable"
                reason = "The event time is not available."
            else:
                status = "measured" if event["status"] in ("measured", "confirmed") else event["status"]
                if status == "manual":
                    status = "confirmed"
                value = nearest_value(times, source, float(event["t"]), max_dt=max(sample_period * 1.5, 0.08))
                if value is None:
                    status = "unavailable"
                    reason = "The joints were not visible at this event."
                elif event["status"] == "candidate":
                    reason = event.get("note") or "Confirm this event before using it as a reference."
        elif spec.reducer == "duration":
            phase = phases.get(spec.phase_id or "")
            if phase is None:
                status = "unavailable"
                reason = "This phase has not been marked."
            else:
                value = float(phase["end_s"]) - float(phase["start_s"])
                if value < -1e-6:
                    value = None
                    status = "unavailable"
                    reason = "The phase end is before its start."
                else:
                    depends = _duration_status(profile, phase["id"], events)
                    status = depends
                    if depends == "candidate":
                        reason = "Confirm the events that bound this phase before using the time as a reference."
        elif spec.reducer == "hip_drop":
            value, status, reason = _hip_drop(rep, times, series, events, sample_period)
        elif spec.reducer == "cadence":
            duration = float(rep["end_s"]) - float(rep["start_s"])
            if duration <= 0.05:
                status = "unavailable"
                reason = "The rep duration is too short to estimate cadence."
            else:
                value = 60.0 / duration
        else:
            status = "unavailable"
            reason = "This measurement is not implemented."
        metrics[spec.id] = _metric_payload(value, spec.unit, status, reason)
    return metrics


def _duration_status(profile: MovementProfile, phase_id: str, events: dict[str, dict]) -> str:
    """Phase time is candidate when either bounding event is still a candidate."""
    if profile.phases is None:
        return "measured"
    mapping = {
        profile.phases.toward: (profile.phases.start_event, profile.phases.extreme_event),
        profile.phases.extreme: (profile.phases.extreme_event, profile.phases.extreme_event),
        profile.phases.returning: (profile.phases.extreme_event, profile.phases.end_event),
    }
    bounds = mapping.get(phase_id)
    if bounds is None:
        return "measured"
    statuses = []
    for event_id in bounds:
        event = events.get(event_id)
        if event is None or event.get("t") is None:
            return "unavailable"
        statuses.append(event["status"])
    if any(status == "candidate" for status in statuses):
        return "candidate"
    if any(status in ("manual", "confirmed") for status in statuses) and all(
        status in ("measured", "confirmed", "manual") for status in statuses
    ):
        return "confirmed" if any(status in ("confirmed", "manual") for status in statuses) else "measured"
    if all(status == "measured" for status in statuses):
        return "measured"
    return "unavailable"


def _hip_drop(rep: dict, times: list[float], series: dict[str, list], events: dict, sample_period: float):
    hip_y = series.get("hip_y") or []
    torso = series.get("torso_len") or []
    knee = series.get("knee") or []
    if not hip_y or not torso:
        return None, "unavailable", "Hip drop needs a visible torso, and it is not a metric distance."
    standing_y = []
    standing_torso = []
    for moment, knee_value, y_value, torso_value in zip(times, knee or [None] * len(times), hip_y, torso):
        if not (rep["start_s"] - 1e-9 <= moment <= rep["end_s"] + 1e-9):
            continue
        if knee_value is not None and knee_value >= 155 and y_value is not None and torso_value is not None:
            standing_y.append(y_value)
            standing_torso.append(torso_value)
    if not standing_y:
        # Fall back to the first visible samples in the rep.
        for moment, y_value, torso_value in zip(times, hip_y, torso):
            if rep["start_s"] - 1e-9 <= moment <= rep["start_s"] + 0.25 and y_value is not None and torso_value is not None:
                standing_y.append(y_value)
                standing_torso.append(torso_value)
    if not standing_y or not standing_torso:
        return None, "unavailable", "The standing hip and torso were not both visible."
    torso_len = sum(standing_torso) / len(standing_torso)
    if torso_len < 8:
        return None, "unavailable", "The torso is too small in frame to use as a scale proxy."
    bottom = events.get("bottom")
    at = float(bottom["t"]) if bottom and bottom.get("t") is not None else rep.get("extreme_s", rep["start_s"])
    bottom_y = nearest_value(times, hip_y, at, max_dt=max(sample_period * 1.5, 0.08))
    if bottom_y is None:
        return None, "unavailable", "The hip was not visible at the bottom."
    stand_y = sum(standing_y) / len(standing_y)
    # Image y grows downward, so a deeper hip is a larger y.
    return (bottom_y - stand_y) / torso_len, "measured", "2D hip drop divided by standing torso length."


def _quality_from_tracking(rep_quality: str, notes: list[str]) -> tuple[str, list[str]]:
    return rep_quality, notes


def proposed_to_rep(profile: MovementProfile, proposed: ProposedRep, index: int, rep_id: str | None = None) -> dict:
    return {
        "id": rep_id or new_id("rep"),
        "index": index,
        "start_s": round(proposed.start_s, 4),
        "end_s": round(proposed.end_s, 4),
        "extreme_s": round(proposed.extreme_s, 4),
        "set_label": None,
        "source": "auto",
        "edited": False,
        "quality": proposed.quality,
        "notes": proposed.notes,
        "phases": proposed.phases,
        "events": proposed.events,
        "missing_fraction": round(proposed.missing_fraction, 4),
        "metrics": {},
    }


def _sample_period(times: list[float]) -> float:
    if len(times) < 2:
        return 1 / 12
    deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    if not deltas:
        return 1 / 12
    deltas.sort()
    return deltas[len(deltas) // 2]


def derive_analysis(
    *,
    profile_id: str,
    times: list[float],
    series: dict[str, list[float | None]],
    meta: dict,
    landmarks: list[dict] | None = None,
    short_gap_s: float = SHORT_GAP_S,
    existing_reps: list[dict] | None = None,
    analysis_id: str | None = None,
) -> dict:
    """Build an analysis dict. ``existing_reps`` keeps manual edits when set."""
    profile = get_profile(profile_id)
    view = meta.get("view") or profile.default_view
    facing = meta.get("facing") or ""
    handedness = meta.get("handedness") or ""
    front_leg = (meta.get("context") or {}).get("front_leg")
    if front_leg not in ("left", "right"):
        front_leg = None
    preference = {}
    if profile.id == "reverse_lunge" and front_leg:
        preference["knee"] = front_leg
    # Throwing and bowling actions use the declared arm even if the other side is clearer.
    strict_arm = profile.id in {"cricket_bowling", "tennis_serve", "free_throw", "baseball_pitch"}
    if strict_arm and handedness in ("left", "right"):
        preference["elbow"] = handedness
        preference["wrist"] = handedness
        preference["knee"] = "left" if handedness == "right" else "right"
    elif profile.id == "reverse_lunge" and front_leg:
        preference["knee"] = front_leg
    sides = {
        "elbow": choose_side(
            series.get("elbow_left"), series.get("elbow_right"), preference.get("elbow"), strict=strict_arm
        ),
        "knee": choose_side(
            series.get("knee_left"), series.get("knee_right"), preference.get("knee"), strict=strict_arm
        ),
        "hip": choose_side(series.get("hip_left"), series.get("hip_right"), preference.get("hip")),
        "wrist": preference.get("wrist")
        if strict_arm and preference.get("wrist")
        else choose_side(series.get("wrist_speed_left"), series.get("wrist_speed_right"), preference.get("wrist")),
    }
    if profile.id == "reverse_lunge" and not front_leg:
        # The chosen leg is a guess. Reps are uncertain until the user sets the front leg.
        sides["knee_guess"] = True
    full = _alias_series(series, sides)
    signal_name = profile.segmenter.signal
    signal = full.get(signal_name or "", [])
    if existing_reps is None:
        proposed = segment_signal(profile, times, signal) if signal_name else []
        if profile.id == "deadlift":
            for item in proposed:
                apply_knee_pass(profile, item, times, full.get("knee"))
        if profile.id == "cricket_bowling":
            for item in proposed:
                stride = delivery_stride_time(times, full.get("knee"), item.start_s, item.extreme_s)
                for event in item.events:
                    if event["id"] == "delivery_stride" and stride is not None:
                        event["t"] = stride
                        event["status"] = "candidate"
        reps = [proposed_to_rep(profile, item, index + 1) for index, item in enumerate(proposed)]
        if profile.id == "reverse_lunge" and not front_leg:
            for rep in reps:
                if rep["quality"] == "ok":
                    rep["quality"] = "uncertain"
                    rep["notes"] = rep["notes"] + ["Front leg was chosen from range of motion. Set the front leg to confirm it."]
    else:
        reps = existing_reps
    period = _sample_period(times)
    for rep in reps:
        rep["metrics"] = compute_metrics(profile, rep, times, full, view, period)
        rep["sides"] = {key: value for key, value in sides.items() if key != "knee_guess"}
    plot_series = {}
    for name, values in full.items():
        if name.startswith("wrist_speed"):
            continue
        plotted, filled = interpolate_short_gaps(values, times, short_gap_s)
        plot_series[name] = {"y": [None if value is None else round(float(value), 4) for value in plotted], "interpolated": filled}
    tracked = 0
    if landmarks is not None:
        tracked = sum(1 for frame in landmarks if frame.get("lm"))
        total = len(landmarks)
    else:
        knee = full.get("knee") or full.get("elbow") or []
        total = len(times)
        tracked = sum(1 for value in knee if value is not None)
    warnings = list(meta.get("warnings") or [])
    if total and tracked / total < 0.3:
        warnings.append("Pose tracking covered less than 30% of the analysed frames. Missing measurements are left empty.")
    if not reps and profile.segmenter.mode == "manual":
        warnings.append("Automatic segmentation is not available for this movement. Mark the rep manually.")
    elif not reps and profile.segmenter.mode != "manual":
        warnings.append("No rep was detected. You can mark the boundaries manually.")
    view_ok = any(key == view for key, _label in profile.views)
    if not view_ok:
        warnings.append("This camera view is outside the views listed for the movement.")
    return {
        "id": analysis_id or new_id("an"),
        "profile_id": profile.id,
        "profile_version": profile.version,
        "clip_id": meta.get("clip_id"),
        "filename": meta.get("filename") or "",
        "variant": meta.get("variant") or profile.default_variant,
        "view": view,
        "facing": facing,
        "handedness": handedness,
        "context": dict(meta.get("context") or {}),
        "athlete": meta.get("athlete") or "Athlete",
        "recorded_at": meta.get("recorded_at"),
        "created_at": meta.get("created_at"),
        "synthetic": bool(meta.get("synthetic")),
        "source_kind": meta.get("source_kind") or ("synthetic_fixture" if meta.get("synthetic") else "upload"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "timing_mode": meta.get("timing_mode") or "unknown",
        "fps": meta.get("fps"),
        "sample_fps": meta.get("sample_fps"),
        "times": [round(float(value), 4) for value in times],
        "series": {
            name: [None if value is None else round(float(value), 4) for value in values]
            for name, values in full.items()
            if not name.startswith("wrist_speed") or profile.segmenter.mode == "peak"
        },
        "plot_series": plot_series,
        "landmarks": landmarks,
        "sides": {key: value for key, value in sides.items() if key != "knee_guess"},
        "reps": reps,
        "warnings": warnings,
        "tracking_fraction": round(tracked / total, 4) if total else 0.0,
        "params": {
            "short_gap_s": short_gap_s,
            "visibility_min": meta.get("visibility_min", VISIBILITY_MIN),
            "seg_version": meta.get("seg_version"),
        },
    }


def blank_manual_rep(profile: MovementProfile, start_s: float, end_s: float, index: int = 1) -> dict:
    events = []
    for spec in profile.events:
        events.append(
            {
                "id": spec.id,
                "label": spec.label,
                "t": None,
                "status": "unavailable",
                "source": "manual",
                "note": spec.note,
            }
        )
    phases = []
    if profile.phases is not None:
        phases = [
            {"id": profile.phases.toward, "label": profile.phases.toward_label, "start_s": start_s, "end_s": end_s},
        ]
    return {
        "id": new_id("rep"),
        "index": index,
        "start_s": start_s,
        "end_s": end_s,
        "extreme_s": start_s,
        "set_label": None,
        "source": "manual",
        "edited": True,
        "quality": "uncertain",
        "notes": ["Marked manually."],
        "phases": phases,
        "events": events,
        "missing_fraction": 0.0,
        "metrics": {},
        "sides": {},
    }


def rebuild_phases_from_events(profile: MovementProfile, rep: dict) -> None:
    if profile.phases is None:
        return
    events = _event_lookup(rep)
    start = events.get(profile.phases.start_event)
    extreme = events.get(profile.phases.extreme_event)
    end = events.get(profile.phases.end_event)
    if not start or start.get("t") is None or not extreme or extreme.get("t") is None:
        return
    start_s = float(start["t"])
    extreme_s = float(extreme["t"])
    end_s = float(end["t"]) if end and end.get("t") is not None else float(rep["end_s"])
    rep["phases"] = [
        {
            "id": profile.phases.toward,
            "label": profile.phases.toward_label,
            "start_s": start_s,
            "end_s": extreme_s,
        },
        {
            "id": profile.phases.extreme,
            "label": profile.phases.extreme_label,
            "start_s": extreme_s,
            "end_s": extreme_s,
        },
        {
            "id": profile.phases.returning,
            "label": profile.phases.returning_label,
            "start_s": extreme_s,
            "end_s": end_s,
        },
    ]
    rep["extreme_s"] = extreme_s


def apply_rep_edit(
    analysis: dict,
    rep_id: str,
    *,
    start_s: float,
    end_s: float,
    set_label: str | None,
    event_times: dict[str, float | None] | None = None,
    confirm_ids: set[str] | None = None,
) -> dict:
    """Store a manual correction and recompute metrics from the raw series."""
    profile = get_profile(analysis["profile_id"])
    rep = next(item for item in analysis["reps"] if item["id"] == rep_id)
    if end_s <= start_s:
        raise ValueError("The rep end must be after the start.")
    rep["start_s"] = round(float(start_s), 4)
    rep["end_s"] = round(float(end_s), 4)
    label = (set_label or "").strip()
    rep["set_label"] = label or None
    rep["edited"] = True
    rep["source"] = "manual" if rep.get("source") == "manual" else rep.get("source", "auto")
    confirm_ids = confirm_ids or set()
    if event_times:
        for event in rep["events"]:
            if event["id"] not in event_times:
                continue
            moment = event_times[event["id"]]
            if moment is None:
                event["t"] = None
                event["status"] = "unavailable"
                continue
            event["t"] = round(float(moment), 4)
            event["source"] = "manual"
            if event["id"] in confirm_ids:
                event["status"] = "confirmed"
            elif event.get("status") == "candidate":
                event["status"] = "candidate"
        rebuild_phases_from_events(profile, rep)
    times = [float(value) for value in analysis["times"]]
    series = {name: values for name, values in analysis["series"].items()}
    period = _sample_period(times)
    rep["metrics"] = compute_metrics(profile, rep, times, series, analysis["view"], period)
    if rep["quality"] == "incomplete" and rep["edited"]:
        rep["quality"] = "uncertain"
        rep["notes"] = list(rep.get("notes") or []) + ["Boundaries were edited manually."]
    return rep


def add_manual_rep(analysis: dict, start_s: float | None = None, end_s: float | None = None) -> dict:
    profile = get_profile(analysis["profile_id"])
    times = [float(value) for value in analysis["times"]]
    start = float(times[0] if start_s is None else start_s)
    end = float(times[-1] if end_s is None else end_s)
    rep = blank_manual_rep(profile, start, end, index=len(analysis["reps"]) + 1)
    rep["sides"] = dict(analysis.get("sides") or {})
    series = {name: values for name, values in analysis["series"].items()}
    rep["metrics"] = compute_metrics(profile, rep, times, series, analysis["view"], _sample_period(times))
    analysis["reps"].append(rep)
    _reindex(analysis)
    return rep


def delete_rep(analysis: dict, rep_id: str) -> None:
    analysis["reps"] = [rep for rep in analysis["reps"] if rep["id"] != rep_id]
    _reindex(analysis)


def resegment(analysis: dict) -> dict:
    """Replace automatic reps. Manually edited reps stay, with their timestamps."""
    profile = get_profile(analysis["profile_id"])
    kept = [rep for rep in analysis["reps"] if rep.get("edited") or rep.get("source") == "manual"]
    fresh = derive_analysis(
        profile_id=profile.id,
        times=[float(value) for value in analysis["times"]],
        series={name: list(values) for name, values in analysis["series"].items() if not name.startswith("wrist_speed") or True},
        meta={
            "view": analysis["view"],
            "facing": analysis.get("facing"),
            "handedness": analysis.get("handedness"),
            "context": analysis.get("context") or {},
            "variant": analysis.get("variant"),
            "clip_id": analysis.get("clip_id"),
            "filename": analysis.get("filename"),
            "athlete": analysis.get("athlete"),
            "recorded_at": analysis.get("recorded_at"),
            "created_at": analysis.get("created_at"),
            "synthetic": analysis.get("synthetic"),
            "source_kind": analysis.get("source_kind"),
            "width": analysis.get("width"),
            "height": analysis.get("height"),
            "timing_mode": analysis.get("timing_mode"),
            "fps": analysis.get("fps"),
            "sample_fps": analysis.get("sample_fps"),
            "warnings": [],
            "seg_version": (analysis.get("params") or {}).get("seg_version"),
        },
        landmarks=None,
        existing_reps=None,
        analysis_id=analysis["id"],
    )
    # derive_analysis with landmarks None recomputes tracking from the signal.
    automatic = []
    for rep in fresh["reps"]:
        if _overlaps_kept(rep, kept):
            continue
        automatic.append(rep)
    analysis["reps"] = _sort_reps(kept + automatic)
    _reindex(analysis)
    analysis["warnings"] = fresh["warnings"]
    return analysis


def _overlaps_kept(rep: dict, kept: list[dict]) -> bool:
    for other in kept:
        latest_start = max(rep["start_s"], other["start_s"])
        earliest_end = min(rep["end_s"], other["end_s"])
        overlap = earliest_end - latest_start
        duration = min(rep["end_s"] - rep["start_s"], other["end_s"] - other["start_s"])
        if duration > 0 and overlap / duration > 0.5:
            return True
    return False


def _sort_reps(reps: list[dict]) -> list[dict]:
    return sorted(reps, key=lambda rep: (rep["start_s"], rep["end_s"]))


def _reindex(analysis: dict) -> None:
    for index, rep in enumerate(_sort_reps(analysis["reps"]), start=1):
        rep["index"] = index
    analysis["reps"] = _sort_reps(analysis["reps"])


def phase_resample(
    times: list[float],
    values: list[float | None],
    phases: list[dict],
    per_phase: int = 16,
    short_gap_s: float = SHORT_GAP_S,
) -> dict:
    """Equal-width phase samples for shape comparison.

    The x positions are phase progress, not seconds. Timing metrics must keep
    using the original timestamps.
    """
    from primed.series import sample_short_gap

    xs: list[float] = []
    ys: list[float | None] = []
    labels: list[str] = []
    source_t: list[float] = []
    phase_ids: list[str] = []
    usable_phases = [phase for phase in phases if phase["end_s"] >= phase["start_s"]]
    for phase_index, phase in enumerate(usable_phases):
        span = float(phase["end_s"]) - float(phase["start_s"])
        for step in range(per_phase):
            fraction = 0.0 if per_phase == 1 else step / (per_phase - 1)
            moment = float(phase["start_s"]) + span * fraction
            xs.append(phase_index + fraction)
            labels.append(phase["label"])
            phase_ids.append(phase["id"])
            source_t.append(round(moment, 4))
            if span <= 1e-6:
                ys.append(nearest_value(times, values, moment, max_dt=short_gap_s))
            else:
                ys.append(sample_short_gap(times, values, moment, short_gap_s))
    return {
        "x": xs,
        "y": [None if value is None else round(float(value), 4) for value in ys],
        "labels": labels,
        "phase_ids": phase_ids,
        "source_t": source_t,
    }


def value_at_time(analysis: dict, series_name: str, moment: float) -> float | None:
    times = [float(value) for value in analysis["times"]]
    values = analysis["series"].get(series_name) or []
    if len(times) < 2:
        return nearest_value(times, values, moment, max_dt=1.0)
    period = _sample_period(times)
    return nearest_value(times, values, moment, max_dt=max(period * 1.25, 0.05))
