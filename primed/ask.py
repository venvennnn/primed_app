"""Ask Primed.

The application resolves the measurement and the evidence. Gemini, when a key
is configured, only receives derived records and the question. Cited ids,
units, values and timestamps are checked against those records. A mismatched
answer is replaced with the deterministic one.

No video is included. The API key is read from the environment and is not
returned, logged, or written into the workspace.
"""

from __future__ import annotations

import json
import re

from primed.baseline import _hard_scope, _rep_change, active_reference
from primed.config import gemini_api_key, gemini_model
from primed.profiles import get_profile

_REP_ID = re.compile(r"rep_[a-f0-9]{12}")


def build_bundle(
    workspace: dict,
    *,
    profile_id: str,
    variant: str | None = None,
    view: str | None = None,
    selected_set: str | None = None,
    selected_load: float | None = None,
    question: str = "",
) -> dict:
    reference = active_reference(workspace, profile_id, variant, view)
    profile = get_profile(profile_id)
    reps = []
    for analysis in workspace.get("analyses", {}).values():
        if analysis.get("profile_id") != profile_id:
            continue
        if reference is not None:
            reason = _hard_scope(analysis, reference)
            if reason:
                continue
        elif variant and analysis.get("variant") != variant:
            continue
        elif view and analysis.get("view") != view:
            continue
        for rep in analysis.get("reps", []):
            if rep.get("quality") in ("tracking_loss", "incomplete"):
                continue
            metrics = {}
            signed = {}
            standardised = {}
            for metric_id, metric in (rep.get("metrics") or {}).items():
                if not metric.get("valid") or metric.get("value") is None:
                    continue
                metrics[metric_id] = {
                    "value": metric["value"],
                    "unit": metric["unit"],
                    "valid": True,
                }
                if reference is not None:
                    change = _rep_change(rep, analysis, reference, metric_id)
                    if change:
                        signed[metric_id] = round(change["signed"], 4)
                        if change["standardised"] is not None:
                            standardised[metric_id] = round(change["standardised"], 4)
            reps.append(
                {
                    "rep_id": rep["id"],
                    "analysis_id": analysis["id"],
                    "clip_id": analysis.get("clip_id"),
                    "profile_id": profile_id,
                    "variant": analysis.get("variant"),
                    "view": analysis.get("view"),
                    "set_label": rep.get("set_label"),
                    "day": (analysis.get("recorded_at") or "")[:10],
                    "load_kg": (analysis.get("context") or {}).get("load_kg"),
                    "start_s": rep.get("start_s"),
                    "end_s": rep.get("end_s"),
                    "timestamp_s": rep.get("extreme_s") or rep.get("start_s"),
                    "quality": rep.get("quality"),
                    "synthetic": bool(analysis.get("synthetic")),
                    "metrics": metrics,
                    "signed": signed,
                    "standardised": standardised,
                }
            )
    return {
        "question": question,
        "profile_id": profile_id,
        "profile_name": profile.name,
        "selected_set": selected_set,
        "selected_load": selected_load,
        "reference": None
        if reference is None
        else {
            "id": reference["id"],
            "version": reference["version"],
            "provenance": reference.get("provenance"),
            "n_reps": len(reference.get("source_rep_ids") or []),
            "scalars": {
                metric_id: {
                    "value": scalar.get("value"),
                    "unit": scalar.get("unit"),
                    "n": scalar.get("n"),
                    "std": scalar.get("std"),
                    "sigma_ok": scalar.get("sigma_ok"),
                }
                for metric_id, scalar in (reference.get("scalars") or {}).items()
            },
        },
        "metrics": [{"id": spec.id, "label": spec.label, "unit": spec.unit} for spec in profile.metrics],
        "reps": reps,
        "note": "Movement measurements only. A deviation is not a cause and it is not advice about what to do.",
    }


def _metric_from_question(question: str, catalog: list[dict]) -> str | None:
    text = question.lower()
    best = None
    for item in catalog:
        label = item["label"].lower()
        if label in text or item["id"].replace("_", " ") in text:
            if best is None or len(label) > len(best[0]):
                best = (label, item["id"])
    return None if best is None else best[1]


def _biggest(reps: list[dict], metric_id: str | None, notes: list[str], bundle: dict) -> dict:
    if bundle.get("reference") is None:
        notes.append("There is no baseline yet, so there is no change to compare.")
        evidence = [rep["rep_id"] for rep in reps[:5]]
        return _pack(
            "No baseline is saved for this movement, so this is a movement-only list of the measured reps.",
            [],
            evidence,
            notes,
            "deterministic",
        )
    best = None
    for rep in reps:
        signed = rep.get("signed") or {}
        keys = [metric_id] if metric_id else list(signed)
        for key in keys:
            if key not in signed:
                continue
            magnitude = abs(float(signed[key]))
            if best is None or magnitude > best[0]:
                best = (magnitude, rep, key, float(signed[key]))
    if best is None:
        return _pack(
            "None of the comparable reps have a valid change for this question.",
            [],
            [],
            notes,
            "deterministic",
        )
    _magnitude, rep, key, signed_value = best
    label = _label(bundle, key)
    unit = _unit(bundle, key)
    observed = rep["metrics"][key]["value"]
    answer = (
        f"{label} changed the most on {rep['rep_id']}: {signed_value:+.2f} {unit} "
        f"from the reference (measured {observed:.2f} {unit}). "
        "This is the size of the difference, not a judgement of the movement."
    )
    finding = {
        "rep_id": rep["rep_id"],
        "analysis_id": rep["analysis_id"],
        "metric_id": key,
        "value": round(signed_value, 4),
        "unit": unit,
        "timestamp_s": rep.get("timestamp_s"),
    }
    return _pack(answer, [finding], [rep["rep_id"]], notes, "deterministic")


def _repeats(question: str, reps: list[dict], bundle: dict, metric_id: str | None, notes: list[str]) -> dict:
    rows = reps
    if "same load" in question and bundle.get("selected_load") is not None:
        load = float(bundle["selected_load"])
        rows = [rep for rep in rows if rep.get("load_kg") is not None and float(rep["load_kg"]) == load]
    if metric_id is None:
        metric_id = _busiest_metric(rows)
    if metric_id is None or bundle.get("reference") is None:
        return _pack(
            "There is not a baseline and a measured metric to compare across days. This stays a movement-only review.",
            [],
            [rep["rep_id"] for rep in rows[:3]],
            notes,
            "deterministic",
        )
    by_day: dict[str, list[float]] = {}
    evidence = []
    for rep in rows:
        signed = (rep.get("signed") or {}).get(metric_id)
        if signed is None or not rep.get("day"):
            continue
        by_day.setdefault(rep["day"], []).append(float(signed))
        evidence.append(rep["rep_id"])
    if len(by_day) < 2 or len(evidence) < 3:
        label = _label(bundle, metric_id)
        return _pack(
            f"There are not enough comparable sessions to say the change in {label} repeats. "
            f"Sample size is {len(evidence)} rep(s) on {len(by_day)} day(s). This is a movement-only review.",
            [],
            evidence[:5],
            notes,
            "deterministic",
        )
    day_signs = []
    for day, values in sorted(by_day.items()):
        mean = sum(values) / len(values)
        day_signs.append((day, mean, len(values)))
    positive = sum(1 for _day, mean, _n in day_signs if mean > 0)
    negative = sum(1 for _day, mean, _n in day_signs if mean < 0)
    direction = "higher than the reference" if positive >= negative else "lower than the reference"
    label = _label(bundle, metric_id)
    unit = _unit(bundle, metric_id)
    answer = (
        f"On {len(day_signs)} days, the mean signed change in {label} was {direction}. "
        f"Sample size is {len(evidence)} reps. "
        + ", ".join(f"{day}: {mean:+.2f} {unit} (n={count})" for day, mean, count in day_signs)
        + ". This describes the measured records. It is not a cause and it is not advice."
    )
    return _pack(answer, [], evidence[:8], notes, "deterministic")


def _summary(reps: list[dict], bundle: dict, notes: list[str]) -> dict:
    if not reps:
        return _pack("There are no comparable reps for this movement yet.", [], [], notes, "deterministic")
    text = f"{bundle.get('profile_name')} has {len(reps)} comparable rep(s)."
    if bundle.get("reference"):
        text += f" The active reference is version {bundle['reference']['version']}."
    else:
        text += " No baseline is saved yet."
    return _pack(text, [], [rep["rep_id"] for rep in reps[:5]], notes, "deterministic")


def answer_question(question: str, bundle: dict) -> dict:
    text = (question or "").lower()
    notes: list[str] = []
    reps = list(bundle.get("reps") or [])
    if "this set" in text:
        selected = bundle.get("selected_set")
        if selected:
            reps = [rep for rep in reps if rep.get("set_label") == selected]
        else:
            notes.append("No set is selected, so this uses every comparable rep.")
    metric_id = _metric_from_question(text, bundle.get("metrics") or [])
    if any(phrase in text for phrase in ("other day", "other days", "same load", "repeat")):
        return _repeats(text, reps, bundle, metric_id, notes)
    if any(phrase in text for phrase in ("which rep", "biggest", "changed the most", "most")):
        return _biggest(reps, metric_id, notes, bundle)
    return _summary(reps, bundle, notes)


def validate_model_answer(raw: dict, bundle: dict) -> dict:
    """Drop findings that do not match the supplied measurements."""
    known = {rep["rep_id"]: rep for rep in bundle.get("reps") or []}
    notes = list(raw.get("notes") or [])
    findings = []
    for item in raw.get("findings") or []:
        if not isinstance(item, dict):
            notes.append("Removed a finding that was not a measurement.")
            continue
        rep = known.get(item.get("rep_id"))
        if rep is None:
            notes.append("Removed a finding that cited an unknown rep.")
            continue
        metric_id = item.get("metric_id")
        metric = (rep.get("metrics") or {}).get(metric_id)
        if not metric:
            notes.append("Removed a finding that cited an unknown measurement.")
            continue
        unit = item.get("unit")
        if unit not in (metric.get("unit"), "σ"):
            notes.append("Removed a finding whose unit does not match the measurement.")
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            notes.append("Removed a finding without a numeric value.")
            continue
        allowed = [float(metric["value"])]
        if metric_id in (rep.get("signed") or {}):
            allowed.append(float(rep["signed"][metric_id]))
        if metric_id in (rep.get("standardised") or {}):
            allowed.append(float(rep["standardised"][metric_id]))
        if not any(abs(value - candidate) <= 0.2 for candidate in allowed):
            notes.append("Removed a finding whose value is not in the measured records.")
            continue
        timestamp = item.get("timestamp_s")
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                notes.append("Removed a finding with a bad timestamp.")
                continue
            start = float(rep.get("start_s") or 0)
            end = float(rep.get("end_s") or start)
            if timestamp < start - 0.05 or timestamp > end + 0.05:
                notes.append("Removed a finding whose timestamp is outside the rep.")
                continue
        findings.append(
            {
                "rep_id": rep["rep_id"],
                "analysis_id": rep.get("analysis_id") or "",
                "metric_id": metric_id,
                "value": round(value, 4),
                "unit": unit,
                "timestamp_s": timestamp,
            }
        )
    evidence = [rep_id for rep_id in raw.get("evidence_rep_ids") or [] if rep_id in known]
    answer = str(raw.get("answer") or "")
    unknown = set(_REP_ID.findall(answer)) - set(known)
    for rep_id in unknown:
        answer = answer.replace(rep_id, "an unknown rep")
    if unknown:
        notes.append("Removed rep ids that are not in the measured records.")
    if not findings and (raw.get("findings") or unknown):
        fallback = answer_question(bundle.get("question") or "", bundle)
        fallback["notes"] = notes + list(fallback.get("notes") or [])
        fallback["notes"].append("The model response did not match the stored measurements, so this answer uses the measured records.")
        fallback["source"] = "deterministic"
        return fallback
    if not evidence and findings:
        evidence = [findings[0]["rep_id"]]
    return {
        "answer": answer,
        "findings": findings,
        "evidence_rep_ids": evidence,
        "notes": notes,
        "source": "gemini",
    }


def ask_gemini(question: str, bundle: dict) -> dict:
    key = gemini_api_key()
    if not key:
        result = answer_question(question, bundle)
        result.setdefault("notes", []).append("Ask Primed is using the measured records because no Gemini API key is configured.")
        return result
    payload = dict(bundle)
    payload["question"] = question
    try:
        from google import genai
        from google.genai import types
        from pydantic import BaseModel, Field

        class Finding(BaseModel):
            rep_id: str
            analysis_id: str = ""
            metric_id: str
            value: float
            unit: str
            timestamp_s: float | None = None

        class AskAnswer(BaseModel):
            answer: str = Field(description="Short answer using only the supplied measurements.")
            findings: list[Finding]
            evidence_rep_ids: list[str]

        client = genai.Client(api_key=key)
        prompt = (
            "Answer the athlete or coach using only the JSON measurement records. "
            "Do not invent rep ids, metric names, units, values, or timestamps. "
            "Do not claim a cause, an improvement, or a coaching recommendation. "
            "If the records are not enough, say so.\n\n"
            f"Question: {question}\n\nRecords:\n{json.dumps(payload)}"
        )
        response = client.models.generate_content(
            model=gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_json_schema=AskAnswer.model_json_schema(),
            ),
        )
        parsed = json.loads(response.text or "{}")
        parsed["question"] = question
        return validate_model_answer(parsed, payload)
    except Exception:
        result = answer_question(question, bundle)
        result.setdefault("notes", []).append("The Gemini request failed. Showing the measured answer instead.")
        result["source"] = "deterministic"
        return result


def _pack(answer: str, findings: list[dict], evidence: list[str], notes: list[str], source: str) -> dict:
    return {
        "answer": answer,
        "findings": findings,
        "evidence_rep_ids": evidence,
        "notes": [note for note in notes if note],
        "source": source,
    }


def _label(bundle: dict, metric_id: str) -> str:
    for item in bundle.get("metrics") or []:
        if item["id"] == metric_id:
            return item["label"]
    return metric_id


def _unit(bundle: dict, metric_id: str) -> str:
    for item in bundle.get("metrics") or []:
        if item["id"] == metric_id:
            return item["unit"]
    return ""


def _busiest_metric(reps: list[dict]) -> str | None:
    scores: dict[str, float] = {}
    for rep in reps:
        for key, value in (rep.get("signed") or {}).items():
            scores[key] = max(scores.get(key, 0.0), abs(float(value)))
    if not scores:
        return None
    return max(scores, key=scores.get)
