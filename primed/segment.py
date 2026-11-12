"""Rep segmentation.

Dip and rise detectors use a documented threshold state machine on one signal.
Peak mode only proposes a candidate window. Manual mode proposes nothing.
Detection may median-smooth the signal. Returned event times still point at
the original clock. Manual edits are applied by the caller and are not overwritten
here.
"""

from __future__ import annotations

from dataclasses import dataclass

from primed.profiles import MovementProfile, PhaseMap
from primed.series import median_smooth


@dataclass
class ProposedRep:
    start_s: float
    end_s: float
    extreme_s: float
    quality: str
    notes: list[str]
    phases: list[dict]
    events: list[dict]
    missing_fraction: float


def _slice_missing(values: list[float | None], start: int, end: int) -> float:
    window = values[start : end + 1]
    if not window:
        return 1.0
    missing = sum(1 for value in window if value is None)
    return missing / len(window)


def _band_bounds(
    times: list[float],
    values: list[float | None],
    start_i: int,
    end_i: int,
    extreme_i: int,
    extreme_v: float,
    band: float,
    mode: str,
) -> tuple[float, float]:
    def inside(value: float | None) -> bool:
        if value is None:
            return False
        if mode == "dip":
            return value <= extreme_v + band
        return value >= extreme_v - band

    left = extreme_i
    while left - 1 >= start_i and inside(values[left - 1]):
        left -= 1
    right = extreme_i
    while right + 1 <= end_i and inside(values[right + 1]):
        right += 1
    return times[left], times[right]


def _phases_for(profile_phases: PhaseMap, start_s: float, band_start: float, band_end: float, end_s: float) -> list[dict]:
    return [
        {
            "id": profile_phases.toward,
            "label": profile_phases.toward_label,
            "start_s": start_s,
            "end_s": band_start,
        },
        {
            "id": profile_phases.extreme,
            "label": profile_phases.extreme_label,
            "start_s": band_start,
            "end_s": band_end,
        },
        {
            "id": profile_phases.returning,
            "label": profile_phases.returning_label,
            "start_s": band_end,
            "end_s": end_s,
        },
    ]


def _events_for(
    profile: MovementProfile,
    start_s: float,
    extreme_s: float,
    end_s: float,
    quality: str,
) -> list[dict]:
    phases = profile.phases
    assert phases is not None
    times = {
        phases.start_event: start_s,
        phases.extreme_event: extreme_s,
        phases.end_event: end_s,
    }
    events = []
    for spec in profile.events:
        moment = times.get(spec.id)
        if moment is None:
            events.append(
                {
                    "id": spec.id,
                    "label": spec.label,
                    "t": None,
                    "status": "unavailable" if spec.kind == "measured" else spec.kind,
                    "source": "auto",
                    "note": spec.note,
                }
            )
            continue
        status = spec.kind
        if spec.kind == "measured" and quality == "incomplete":
            status = "measured"
        events.append(
            {
                "id": spec.id,
                "label": spec.label,
                "t": moment,
                "status": status if status in ("measured", "candidate", "confirmed", "manual") else "measured",
                "source": "auto",
                "note": spec.note,
            }
        )
    return events


def _propose(
    profile: MovementProfile,
    times: list[float],
    raw_values: list[float | None],
    start_i: int,
    end_i: int,
    extreme_i: int,
    quality: str,
    notes: list[str],
) -> ProposedRep:
    assert profile.phases is not None
    start_s = times[start_i]
    end_s = times[end_i]
    extreme_s = times[extreme_i]
    extreme_v = raw_values[extreme_i]
    if extreme_v is None:
        band_start, band_end = extreme_s, extreme_s
    else:
        band_start, band_end = _band_bounds(
            times,
            raw_values,
            start_i,
            end_i,
            extreme_i,
            extreme_v,
            profile.segmenter.band,
            profile.segmenter.mode,
        )
    missing = _slice_missing(raw_values, start_i, end_i)
    if missing > 0.40:
        quality = "tracking_loss"
        notes = notes + ["More than 40% of the signal is missing inside this rep."]
    elif missing > 0.25 and quality == "ok":
        quality = "uncertain"
        notes = notes + ["Part of the signal is missing inside this rep."]
    return ProposedRep(
        start_s=start_s,
        end_s=end_s,
        extreme_s=extreme_s,
        quality=quality,
        notes=notes,
        phases=_phases_for(profile.phases, start_s, band_start, band_end, end_s),
        events=_events_for(profile, start_s, extreme_s, end_s, quality),
        missing_fraction=missing,
    )


def _segment_threshold(
    profile: MovementProfile,
    times: list[float],
    values: list[float | None],
) -> list[ProposedRep]:
    spec = profile.segmenter
    mode = spec.mode
    smoothed = median_smooth(values, 3)
    proposed: list[ProposedRep] = []
    armed: int | None = None
    state = "seek"
    start_i: int | None = None
    extreme_i: int | None = None
    extreme_v: float | None = None
    gap_t: float | None = None

    def is_armed(value: float) -> bool:
        return value >= spec.arm if mode == "dip" else value <= spec.arm

    def is_entering(value: float) -> bool:
        return value < spec.enter if mode == "dip" else value > spec.enter

    def is_recovered(value: float) -> bool:
        return value >= spec.recover if mode == "dip" else value <= spec.recover

    def is_deeper(value: float) -> bool:
        assert extreme_v is not None
        return value < extreme_v if mode == "dip" else value > extreme_v

    def reached_gate() -> bool:
        assert extreme_v is not None
        return extreme_v <= spec.gate if mode == "dip" else extreme_v >= spec.gate

    def prominence(start_index: int) -> float:
        assert extreme_v is not None
        start_value = values[start_index]
        if start_value is None:
            start_value = spec.arm
        if mode == "dip":
            return start_value - extreme_v
        return extreme_v - start_value

    def finish(end_index: int, closed: bool) -> None:
        nonlocal state, armed, start_i, extreme_i, extreme_v
        assert start_i is not None and extreme_i is not None and extreme_v is not None
        duration = times[end_index] - times[start_i]
        notes: list[str] = []
        quality = "ok"
        prom = prominence(start_i)
        if not closed:
            quality = "incomplete"
            notes.append("The signal did not return before the clip or a tracking gap.")
        elif not reached_gate():
            quality = "ignore"
        elif duration < spec.min_duration_s or duration > spec.max_duration_s:
            quality = "incomplete"
            notes.append("The duration is outside the expected range for this movement.")
        elif prom < spec.min_prominence:
            quality = "uncertain"
            notes.append("The range of motion is below the usual detection threshold.")
        if quality != "ignore":
            proposed.append(_propose(profile, times, values, start_i, end_index, extreme_i, quality, notes))
        state = "seek"
        armed = end_index if closed and is_armed(smoothed[end_index] or values[end_index] or 0) else None
        start_i = None
        extreme_i = None
        extreme_v = None

    for index, (moment, smooth) in enumerate(zip(times, smoothed)):
        if smooth is None:
            if state == "in_rep":
                if gap_t is None:
                    gap_t = moment
                elif moment - gap_t > 0.35 and extreme_i is not None and reached_gate():
                    finish(max(index - 1, start_i or index), closed=False)
            continue
        gap_t = None
        if state == "seek":
            if is_armed(smooth):
                armed = index
            elif armed is not None and is_entering(smooth):
                state = "in_rep"
                start_i = armed
                extreme_i = index
                extreme_v = smooth
        elif state == "in_rep":
            if is_deeper(smooth):
                extreme_v = smooth
                extreme_i = index
            if is_recovered(smooth):
                finish(index, closed=True)
    if state == "in_rep" and extreme_i is not None and reached_gate() and start_i is not None:
        finish(len(times) - 1, closed=False)
    return proposed


def _segment_peak(
    profile: MovementProfile,
    times: list[float],
    values: list[float | None],
) -> list[ProposedRep]:
    spec = profile.segmenter
    pairs = [(index, moment, value) for index, (moment, value) in enumerate(zip(times, values)) if value is not None]
    if len(pairs) < 5 or profile.phases is None:
        return []
    magnitudes = sorted(value for _, _, value in pairs)
    median = magnitudes[len(magnitudes) // 2]
    index, moment, peak = max(pairs, key=lambda item: item[2])
    if peak <= 0 or peak < median * 2.5:
        return []
    start_s = max(times[0], moment - spec.pre_s)
    end_s = min(times[-1], moment + spec.post_s)
    if end_s - start_s < 0.30:
        return []
    start_i = min(range(len(times)), key=lambda cursor: abs(times[cursor] - start_s))
    end_i = min(range(len(times)), key=lambda cursor: abs(times[cursor] - end_s))
    missing = _slice_missing(values, start_i, end_i)
    quality = "uncertain"
    notes = ["This window is a candidate around the fastest 2D wrist movement. Confirm it before treating it as the delivery."]
    if missing > 0.40:
        quality = "tracking_loss"
        notes.append("More than 40% of the signal is missing inside this window.")
    phases = _phases_for(profile.phases, start_s, moment, moment, end_s)
    events = []
    for event in profile.events:
        if event.id == spec.peak_event:
            status, when = "candidate", moment
        elif event.id == profile.phases.start_event:
            status, when = "candidate", start_s
        elif event.id == profile.phases.end_event:
            status, when = "candidate", end_s
        else:
            status, when = "unavailable", None
        events.append(
            {
                "id": event.id,
                "label": event.label,
                "t": when,
                "status": status,
                "source": "auto",
                "note": event.note,
            }
        )
    return [
        ProposedRep(
            start_s=start_s,
            end_s=end_s,
            extreme_s=moment,
            quality=quality,
            notes=notes,
            phases=phases,
            events=events,
            missing_fraction=missing,
        )
    ]


def segment_signal(profile: MovementProfile, times: list[float], values: list[float | None]) -> list[ProposedRep]:
    mode = profile.segmenter.mode
    if mode == "manual" or not times:
        return []
    if mode == "peak":
        return _segment_peak(profile, times, values)
    if mode in ("dip", "rise"):
        return _segment_threshold(profile, times, values)
    return []


def apply_knee_pass(
    profile: MovementProfile,
    proposed: ProposedRep,
    times: list[float],
    knee: list[float | None] | None,
) -> None:
    if profile.id != "deadlift" or not knee:
        return
    ascent = next((phase for phase in proposed.phases if phase["id"] == "ascent"), None)
    if ascent is None:
        return
    crossing: float | None = None
    previous: float | None = None
    for moment, value in zip(times, knee):
        if moment < ascent["start_s"] or moment > ascent["end_s"] or value is None:
            previous = value if value is not None else previous
            continue
        if previous is not None and previous < 150 <= value:
            crossing = moment
            break
        previous = value
    for event in proposed.events:
        if event["id"] == "knee_pass":
            if crossing is None:
                event["status"] = "unavailable"
                event["t"] = None
            else:
                event["status"] = "candidate"
                event["t"] = crossing
                event["note"] = "Knee angle crossed 150° during the pull. The bar is not tracked."


def delivery_stride_time(
    times: list[float],
    knee: list[float | None] | None,
    start_s: float,
    end_s: float,
) -> float | None:
    if not knee:
        return None
    best_t: float | None = None
    best_v: float | None = None
    for moment, value in zip(times, knee):
        if moment < start_s or moment > end_s or value is None:
            continue
        if best_v is None or value < best_v:
            best_v = value
            best_t = moment
    return best_t
