"""Reference versions and drift.

A reference is immutable in its measurements. Replacing it creates a new version.
Selecting an older version only changes which stored summary is used.

Signed change is observed minus reference, in the metric's own unit.
Standardised drift is the absolute signed change divided by the sample standard
deviation of the approved reference reps for that metric. Sigma is withheld
unless there are at least the configured number of reference reps (five by
default) and the standard deviation is not effectively zero.

One reference rep supports raw differences. It does not provide a variance estimate.

Reps use each rep's own change. Sets and days use the median signed change on
the raw chart and the median absolute standardised deviation on the sigma chart.
A set or day with no qualifying reps is a gap.

The 2σ and 3σ lines are guides for that reference spread. They are not
confidence intervals and they are not an injury or coaching judgement.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from primed.config import MIN_REFERENCE_REPS, std_epsilon
from primed.ids import new_id
from primed.measure import phase_resample
from primed.profiles import get_profile


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_day(recorded_at: str | None, tz_name: str) -> str | None:
    if not recorded_at:
        return None
    try:
        moment = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        zone = ZoneInfo(tz_name or "UTC")
    except Exception as exc:
        raise ValueError(f"Unknown timezone '{tz_name}'.") from exc
    return moment.astimezone(zone).date().isoformat()


def _scope(ref: dict) -> tuple:
    return (ref["profile_id"], ref["variant"], ref["view"])


def references_for(workspace: dict, profile_id: str, variant: str, view: str) -> list[dict]:
    matched = [
        ref
        for ref in workspace.get("references", [])
        if ref["profile_id"] == profile_id and ref["variant"] == variant and ref["view"] == view
    ]
    return sorted(matched, key=lambda ref: ref["version"])


def active_reference(workspace: dict, profile_id: str, variant: str | None = None, view: str | None = None) -> dict | None:
    candidates = []
    for ref in workspace.get("references", []):
        if ref["profile_id"] != profile_id or not ref.get("active"):
            continue
        if variant is not None and ref["variant"] != variant:
            continue
        if view is not None and ref["view"] != view:
            continue
        candidates.append(ref)
    if not candidates:
        return None
    return sorted(candidates, key=lambda ref: ref["version"])[-1]


def get_reference(workspace: dict, reference_id: str) -> dict | None:
    for ref in workspace.get("references", []):
        if ref["id"] == reference_id:
            return ref
    return None


def _analysis_rep(workspace: dict, rep_id: str) -> tuple[dict, dict]:
    for analysis in workspace.get("analyses", {}).values():
        for rep in analysis.get("reps", []):
            if rep["id"] == rep_id:
                return analysis, rep
    raise KeyError(f"Unknown rep '{rep_id}'.")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sigma_reason(count: int, spread: float | None, unit: str, minimum: int) -> tuple[bool, str]:
    if count < 2:
        return False, "A single reference rep supports raw differences. It does not provide a variance estimate."
    if count < minimum:
        return False, f"Sigma needs at least {minimum} reference reps with this measurement. This version has {count}."
    if spread is None:
        return False, "No spread can be estimated from these reference reps."
    if spread < std_epsilon(unit):
        return False, "Reference reps are effectively identical, so a sigma scale is not shown."
    return True, ""


def set_baseline(
    workspace: dict,
    rep_ids: list[str],
    *,
    provenance: str = "self-selected",
    allow_weak: bool = False,
    minimum_reps: int = MIN_REFERENCE_REPS,
) -> dict:
    if not rep_ids:
        raise ValueError("Choose at least one rep.")
    if provenance not in ("self-selected", "coach-approved", "preset"):
        raise ValueError("Unknown reference provenance.")
    chosen: list[tuple[dict, dict]] = []
    for rep_id in rep_ids:
        chosen.append(_analysis_rep(workspace, rep_id))
    profile_ids = {analysis["profile_id"] for analysis, _rep in chosen}
    variants = {analysis["variant"] for analysis, _rep in chosen}
    views = {analysis["view"] for analysis, _rep in chosen}
    if len(profile_ids) != 1 or len(variants) != 1 or len(views) != 1:
        raise ValueError("A reference has to stay on one movement, variant, and camera view.")
    profile = get_profile(next(iter(profile_ids)))
    for analysis, rep in chosen:
        if rep["quality"] in ("tracking_loss", "incomplete") and not allow_weak:
            raise ValueError(
                "This rep lost tracking or never finished. Correct the boundaries, or explicitly include it anyway."
            )
        if analysis.get("synthetic") != chosen[0][0].get("synthetic"):
            raise ValueError("Synthetic and measured reps cannot share a reference.")
    variant = next(iter(variants))
    view = next(iter(views))
    previous = references_for(workspace, profile.id, variant, view)
    version = (previous[-1]["version"] + 1) if previous else 1
    for ref in previous:
        ref["active"] = False
    scalars: dict[str, dict] = {}
    for spec in profile.metrics:
        values = []
        used = []
        for analysis, rep in chosen:
            metric = rep.get("metrics", {}).get(spec.id)
            if not metric or not metric.get("valid") or metric.get("value") is None:
                continue
            if metric.get("unit") != spec.unit:
                continue
            if analysis.get("facing") and chosen[0][0].get("facing") and spec.series == "trunk_tilt":
                if analysis.get("facing") != chosen[0][0].get("facing"):
                    continue
            values.append(float(metric["value"]))
            used.append(rep["id"])
        spread = statistics.stdev(values) if len(values) >= 2 else None
        sigma_ok, reason = _sigma_reason(len(values), spread, spec.unit, minimum_reps)
        scalars[spec.id] = {
            "value": None if not values else round(_mean(values), 4),
            "unit": spec.unit,
            "n": len(values),
            "std": None if spread is None else round(spread, 4),
            "values": [round(value, 4) for value in values],
            "rep_ids": used,
            "sigma_ok": sigma_ok and bool(values),
            "sigma_reason": "" if sigma_ok and values else (reason or "No usable reference reps for this measurement."),
        }
    traces = {}
    for series_name, label, unit in profile.trace_series:
        traces[series_name] = _centre_trace(chosen, series_name, unit, label)
    first_analysis, first_rep = chosen[0]
    reference = {
        "id": new_id("ref"),
        "version": version,
        "active": True,
        "workspace_id": workspace["workspace_id"],
        "profile_id": profile.id,
        "profile_version": profile.version,
        "variant": variant,
        "view": view,
        "handedness": first_analysis.get("handedness") or "",
        "facing": first_analysis.get("facing") or "",
        "context": dict(first_analysis.get("context") or {}),
        "created_at": _now(),
        "provenance": provenance,
        "approvals": [{"action": provenance, "at": _now()}],
        "source_rep_ids": [rep["id"] for _analysis, rep in chosen],
        "source_analysis_ids": list(dict.fromkeys(analysis["id"] for analysis, _rep in chosen)),
        "source_clip_ids": list(dict.fromkeys(analysis.get("clip_id") for analysis, _rep in chosen if analysis.get("clip_id"))),
        "synthetic": bool(first_analysis.get("synthetic")),
        "scalars": scalars,
        "traces": traces,
        "preview_rep_id": first_rep["id"],
        "method": {
            "scalar": "Mean of the valid measurements on the approved reps.",
            "trace": "Pointwise mean on equal-width phase progress. This is a shape guide, not a shared clock.",
            "signed_change": "Observed value minus the reference value, in the metric's own unit.",
            "standardised": "Absolute signed change divided by the sample standard deviation of the approved reference reps.",
            "sets_and_days": "Median signed change on the raw chart, and median absolute standardised deviation on the sigma chart.",
            "sigma_minimum_reps": minimum_reps,
            "guides": "2σ and 3σ are reference-spread guides, not confidence intervals or injury judgements.",
        },
    }
    workspace.setdefault("references", []).append(reference)
    return reference


def _centre_trace(chosen: list[tuple[dict, dict]], series_name: str, unit: str, label: str) -> dict:
    resampled = []
    for analysis, rep in chosen:
        values = analysis.get("series", {}).get(series_name)
        if not values:
            continue
        sample = phase_resample(analysis["times"], values, rep.get("phases") or [])
        if sample["y"]:
            resampled.append(sample)
    if not resampled:
        return {"label": label, "unit": unit, "x": [], "y": [], "n": [], "phase_ids": [], "labels": []}
    width = min(len(sample["y"]) for sample in resampled)
    centre = []
    counts = []
    for index in range(width):
        column = [sample["y"][index] for sample in resampled if sample["y"][index] is not None]
        counts.append(len(column))
        centre.append(None if not column else round(_mean(column), 4))
    template = resampled[0]
    return {
        "label": label,
        "unit": unit,
        "x": template["x"][:width],
        "y": centre,
        "n": counts,
        "phase_ids": template["phase_ids"][:width],
        "labels": template["labels"][:width],
        "source_note": "Equal-width phase progress. Do not read the x-axis as seconds.",
    }


def select_reference(workspace: dict, reference_id: str) -> dict:
    """Make an existing version active. Does not create a version or change its numbers."""
    target = get_reference(workspace, reference_id)
    if target is None:
        raise KeyError(f"Unknown reference '{reference_id}'.")
    for ref in workspace.get("references", []):
        if _scope(ref) == _scope(target):
            ref["active"] = ref["id"] == reference_id
    return target


def approve_reference(workspace: dict, reference_id: str, action: str = "coach-approved") -> dict:
    target = get_reference(workspace, reference_id)
    if target is None:
        raise KeyError(f"Unknown reference '{reference_id}'.")
    if action not in ("coach-approved", "self-selected"):
        raise ValueError("Unknown approval action.")
    target.setdefault("approvals", []).append({"action": action, "at": _now()})
    target["provenance"] = action
    return target


def _hard_scope(analysis: dict, ref: dict) -> str | None:
    if analysis.get("profile_id") != ref["profile_id"]:
        return "Different movement"
    if analysis.get("variant") != ref["variant"]:
        return "Different variant"
    if analysis.get("view") != ref["view"]:
        return "Different camera view"
    profile = get_profile(ref["profile_id"])
    if profile.handedness_required and ref.get("handedness") and analysis.get("handedness"):
        if analysis.get("handedness") != ref.get("handedness"):
            return "Different handedness"
    if bool(analysis.get("synthetic")) != bool(ref.get("synthetic")):
        return "Synthetic and measured sessions are kept separate"
    return None


def signed_change(observed: float, reference: float) -> float:
    return observed - reference


def standardised_deviation(observed: float, reference: float, spread: float | None, sigma_ok: bool) -> float | None:
    if not sigma_ok or spread is None or spread <= 0:
        return None
    return abs(observed - reference) / spread


def _rep_change(rep: dict, analysis: dict, ref: dict, metric_id: str) -> dict | None:
    scalar = ref["scalars"].get(metric_id)
    metric = rep.get("metrics", {}).get(metric_id)
    if scalar is None or scalar.get("value") is None or metric is None or metric.get("value") is None or not metric.get("valid"):
        return None
    if metric.get("unit") != scalar.get("unit"):
        return None
    profile = get_profile(ref["profile_id"])
    spec = profile.metric(metric_id)
    if spec and spec.series == "trunk_tilt" and ref.get("facing") and analysis.get("facing"):
        if ref["facing"] != analysis["facing"]:
            return None
    observed = float(metric["value"])
    reference = float(scalar["value"])
    signed = signed_change(observed, reference)
    sigma = standardised_deviation(observed, reference, scalar.get("std"), bool(scalar.get("sigma_ok")))
    return {
        "observed": observed,
        "reference": reference,
        "signed": signed,
        "standardised": sigma,
        "unit": scalar["unit"],
    }


def build_trend(
    workspace: dict,
    reference: dict,
    *,
    mode: str,
    standardised: bool,
    athlete: str | None = None,
    load: float | None = None,
    timezone_name: str = "UTC",
) -> dict:
    if mode not in ("reps", "sets", "days"):
        raise ValueError("Grouping must be reps, sets, or days.")
    profile = get_profile(reference["profile_id"])
    excluded: list[dict] = []
    scoped: list[tuple[dict, dict]] = []
    for analysis in workspace.get("analyses", {}).values():
        if athlete and analysis.get("athlete") != athlete:
            excluded.append({"rep_id": None, "analysis_id": analysis["id"], "reason": "Different athlete"})
            continue
        if load is not None:
            context_load = (analysis.get("context") or {}).get("load_kg")
            if context_load is None or float(context_load) != float(load):
                for rep in analysis.get("reps", []):
                    excluded.append({"rep_id": rep["id"], "analysis_id": analysis["id"], "reason": "Different load"})
                continue
        for rep in analysis.get("reps", []):
            reason = _hard_scope(analysis, reference)
            if reason:
                excluded.append({"rep_id": rep["id"], "analysis_id": analysis["id"], "reason": reason})
                continue
            scoped.append((analysis, rep))
    scoped.sort(key=lambda pair: (pair[0].get("recorded_at") or "", pair[1]["start_s"], pair[1]["index"]))
    groups = _groups(scoped, mode, timezone_name, excluded)
    series = []
    for spec in profile.metrics:
        scalar = reference["scalars"].get(spec.id) or {}
        points = []
        covered = 0
        considered = 0
        for group in groups:
            qualifying = []
            for analysis, rep in group["members"]:
                considered += 1
                if rep["quality"] in ("tracking_loss", "incomplete"):
                    if not any(item["rep_id"] == rep["id"] and item["reason"] == rep["quality"] for item in excluded):
                        excluded.append(
                            {
                                "rep_id": rep["id"],
                                "analysis_id": analysis["id"],
                                "reason": "Tracking loss" if rep["quality"] == "tracking_loss" else "Incomplete rep",
                            }
                        )
                    continue
                change = _rep_change(rep, analysis, reference, spec.id)
                if change is None:
                    continue
                if standardised and change["standardised"] is None:
                    continue
                qualifying.append((analysis, rep, change))
            covered += len(qualifying)
            if not qualifying:
                points.append(
                    {
                        "x": group["label"],
                        "y": None,
                        "n": 0,
                        "rep_ids": [rep["id"] for _analysis, rep in group["members"]],
                        "hover": f"{group['label']}: no qualifying reps",
                        "spread": None,
                    }
                )
                continue
            if mode == "reps":
                _analysis, rep, change = qualifying[0]
                value = change["standardised"] if standardised else change["signed"]
                spread = None
                hover = (
                    f"{group['label']}<br>{spec.label}: {change['observed']:.2f} {spec.unit}"
                    f"<br>Signed change: {change['signed']:+.2f} {spec.unit}"
                )
                if change["standardised"] is not None:
                    hover += f"<br>Absolute standardised: {change['standardised']:.2f}σ"
                rep_ids = [rep["id"]]
                count = 1
            else:
                if standardised:
                    values = [change["standardised"] for _a, _r, change in qualifying if change["standardised"] is not None]
                else:
                    values = [change["signed"] for _a, _r, change in qualifying]
                value = None if not values else statistics.median(values)
                spread = None if len(values) < 2 else max(values) - min(values)
                count = len(values)
                rep_ids = [rep["id"] for _a, rep, _c in qualifying]
                kind = "median absolute standardised deviation" if standardised else "median signed change"
                hover = f"{group['label']}<br>{kind}: {value:.2f}" if value is not None else group["label"]
                hover += f"<br>{count} reps"
                if spread is not None:
                    hover += f"<br>Spread: {spread:.2f}"
            points.append(
                {
                    "x": group["label"],
                    "y": None if value is None else round(float(value), 4),
                    "n": count,
                    "rep_ids": rep_ids,
                    "hover": hover,
                    "spread": None if spread is None else round(float(spread), 4),
                }
            )
        reason = None
        if standardised and not scalar.get("sigma_ok"):
            reason = scalar.get("sigma_reason") or "Standardised view is unavailable for this measurement."
            for point in points:
                point["y"] = None
        series.append(
            {
                "metric_id": spec.id,
                "label": spec.label,
                "unit": "σ" if standardised and scalar.get("sigma_ok") else spec.unit,
                "raw_unit": spec.unit,
                "points": points,
                "coverage": f"{covered} of {len(scoped)} reps" if scoped else "0 reps",
                "unavailable_reason": reason,
                "ref_n": scalar.get("n"),
                "ref_value": scalar.get("value"),
                "ref_std": scalar.get("std"),
                "sigma_ok": bool(scalar.get("sigma_ok")),
                "sigma_reason": scalar.get("sigma_reason") or "",
            }
        )
    return {
        "reference_id": reference["id"],
        "reference_version": reference["version"],
        "provenance": reference.get("provenance"),
        "mode": mode,
        "standardised": standardised,
        "grouping": _grouping_text(mode, timezone_name),
        "series": series,
        "excluded": excluded,
        "group_count": len(groups),
        "scoped_reps": len(scoped),
        "synthetic": bool(reference.get("synthetic")),
    }


def _groups(scoped, mode: str, timezone_name: str, excluded: list[dict]) -> list[dict]:
    if mode == "reps":
        groups = []
        for index, (analysis, rep) in enumerate(scoped, start=1):
            groups.append({"label": f"Rep {index}", "members": [(analysis, rep)], "sort": index})
        return groups
    buckets: dict[str, dict] = {}
    for analysis, rep in scoped:
        if mode == "sets":
            label = rep.get("set_label")
            if not label:
                excluded.append({"rep_id": rep["id"], "analysis_id": analysis["id"], "reason": "No confirmed set label"})
                continue
            key = str(label)
        else:
            day = local_day(analysis.get("recorded_at"), timezone_name)
            if day is None:
                excluded.append({"rep_id": rep["id"], "analysis_id": analysis["id"], "reason": "No session date"})
                continue
            key = day
        buckets.setdefault(key, {"label": key, "members": [], "sort": analysis.get("recorded_at") or key})
        buckets[key]["members"].append((analysis, rep))
    return [buckets[key] for key in sorted(buckets, key=lambda item: (buckets[item]["sort"], item))]


def _grouping_text(mode: str, timezone_name: str) -> str:
    if mode == "reps":
        return "Each point is one rep, in session order."
    if mode == "sets":
        return "Each point is the median of reps that share a set label you confirmed. Reps without a set label are left out."
    return f"Each point is the median of reps on a calendar day in {timezone_name}."


def compare_rep_traces(analysis: dict, rep: dict, reference: dict | None) -> list[dict]:
    """Phase-aligned shape traces. Source timestamps stay on ``source_t`` and are not stretched."""
    profile = get_profile(analysis["profile_id"])
    charts = []
    for series_name, label, unit in profile.trace_series:
        values = analysis.get("series", {}).get(series_name)
        if not values:
            continue
        current = phase_resample(analysis["times"], values, rep.get("phases") or [])
        reference_trace = None
        if reference and _hard_scope(analysis, reference) is None:
            reference_trace = (reference.get("traces") or {}).get(series_name)
        charts.append(
            {
                "series": series_name,
                "label": label,
                "unit": unit,
                "current": current,
                "reference": reference_trace,
                "note": "Phase progress aligns shape only. Timing uses the original timestamps.",
            }
        )
    return charts
