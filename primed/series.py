"""Series helpers.

Raw observations stay untouched. Plot series may bridge gaps of at most
``max_gap_s``. Longer gaps stay missing. Scalar metrics never read the bridged
plot series.
"""

from __future__ import annotations


def interpolate_short_gaps(
    values: list[float | None],
    times: list[float],
    max_gap_s: float,
) -> tuple[list[float | None], list[bool]]:
    """Return plot values and a mask that is true where a value was filled."""
    plotted = list(values)
    filled = [False] * len(values)
    index = 0
    count = len(values)
    while index < count:
        if plotted[index] is not None:
            index += 1
            continue
        start = index
        while index < count and plotted[index] is None:
            index += 1
        left = start - 1
        right = index
        if left < 0 or right >= count:
            continue
        if times[right] - times[left] > max_gap_s:
            continue
        span = times[right] - times[left]
        left_value = plotted[left]
        right_value = plotted[right]
        if left_value is None or right_value is None or span <= 0:
            continue
        for cursor in range(start, right):
            alpha = (times[cursor] - times[left]) / span
            plotted[cursor] = left_value + (right_value - left_value) * alpha
            filled[cursor] = True
    return plotted, filled


def nearest_value(
    times: list[float],
    values: list[float | None],
    at_time: float,
    max_dt: float,
) -> float | None:
    best: float | None = None
    best_dt = max_dt
    for time, value in zip(times, values):
        if value is None:
            continue
        delta = abs(time - at_time)
        if delta <= best_dt:
            best_dt = delta
            best = value
    return best


def sample_short_gap(
    times: list[float],
    values: list[float | None],
    at_time: float,
    max_gap_s: float,
) -> float | None:
    """Linear sample. Returns None when the surrounding gap is too long."""
    if not times:
        return None
    if at_time < times[0] or at_time > times[-1]:
        return nearest_value(times, values, at_time, max_dt=1e-6)
    right = 0
    while right < len(times) and times[right] < at_time:
        right += 1
    if right == 0:
        return values[0]
    if right >= len(times):
        return values[-1]
    left = right - 1
    if times[right] - times[left] > max_gap_s:
        return None
    left_value = values[left]
    right_value = values[right]
    if left_value is None or right_value is None:
        return None
    span = times[right] - times[left]
    if span <= 0:
        return left_value
    alpha = (at_time - times[left]) / span
    return left_value + (right_value - left_value) * alpha


def median_smooth(values: list[float | None], window: int = 3) -> list[float | None]:
    """Odd-window median used only to detect reps. None stays None."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    half = window // 2
    smoothed: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue
        samples = [
            values[cursor]
            for cursor in range(index - half, index + half + 1)
            if 0 <= cursor < len(values) and values[cursor] is not None
        ]
        samples.sort()
        smoothed.append(samples[len(samples) // 2])
    return smoothed


def finite(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]
