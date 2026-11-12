"""Workspace records, export, and import.

The public demo stores one JSON document per browser session. Raw media stays
in that session folder and is not written into the export. A clip keeps its id,
hash and timestamps so the original file can be reattached later.
"""

from __future__ import annotations

from datetime import datetime, timezone

from primed.config import EXPORT_VERSION
from primed.ids import new_id
from primed.profiles import get_profile


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_workspace(name: str = "My workspace", athlete: str = "Athlete", timezone_name: str = "UTC") -> dict:
    return {
        "export_version": EXPORT_VERSION,
        "workspace_id": new_id("ws"),
        "name": name,
        "athlete": athlete,
        "timezone": timezone_name,
        "created_at": _now(),
        "clips": {},
        "analyses": {},
        "references": [],
        "synthetic": False,
    }


def export_workspace(
    workspace: dict,
    *,
    analysis_ids: list[str] | None = None,
    include_pose: bool = True,
) -> dict:
    """Versioned JSON. Media paths are removed. Derived results and references stay."""
    selected = set(analysis_ids) if analysis_ids is not None else set(workspace.get("analyses", {}))
    analyses = {}
    clip_ids = set()
    rep_ids = set()
    for analysis_id, analysis in workspace.get("analyses", {}).items():
        if analysis_id not in selected:
            continue
        copied = _copy(analysis)
        if not include_pose:
            copied["landmarks"] = None
        analyses[analysis_id] = copied
        if copied.get("clip_id"):
            clip_ids.add(copied["clip_id"])
        for rep in copied.get("reps", []):
            rep_ids.add(rep["id"])
    clips = {}
    for clip_id, clip in workspace.get("clips", {}).items():
        if analysis_ids is not None and clip_id not in clip_ids:
            continue
        copied = {
            "id": clip.get("id", clip_id),
            "filename": _display_name(clip.get("filename") or "clip"),
            "sha256": clip.get("sha256"),
            "duration_s": clip.get("duration_s"),
            "width": clip.get("width"),
            "height": clip.get("height"),
            "fps": clip.get("fps"),
            "timing_mode": clip.get("timing_mode"),
            "created_at": clip.get("created_at"),
            "recorded_at": clip.get("recorded_at"),
            "profile_id": clip.get("profile_id"),
            "variant": clip.get("variant"),
            "view": clip.get("view"),
            "handedness": clip.get("handedness"),
            "facing": clip.get("facing"),
            "context": clip.get("context") or {},
            "athlete": clip.get("athlete"),
            "media_present": False,
            "bytes": clip.get("bytes"),
        }
        clips[clip_id] = copied
    references = []
    for ref in workspace.get("references", []):
        if analysis_ids is not None and not any(rep_id in rep_ids for rep_id in ref.get("source_rep_ids", [])):
            continue
        references.append(_copy(ref))
    return {
        "export_version": EXPORT_VERSION,
        "exported_at": _now(),
        "workspace_id": workspace["workspace_id"],
        "name": workspace.get("name") or "My workspace",
        "athlete": workspace.get("athlete") or "Athlete",
        "timezone": workspace.get("timezone") or "UTC",
        "created_at": workspace.get("created_at"),
        "synthetic": bool(workspace.get("synthetic")),
        "clips": clips,
        "analyses": analyses,
        "references": references,
        "media_included": False,
    }


def import_workspace(payload: dict) -> dict:
    """Validate an export and return a workspace whose media must be reattached."""
    if not isinstance(payload, dict):
        raise ValueError("This file is not a Primed workspace export.")
    if payload.get("export_version") != EXPORT_VERSION:
        raise ValueError("This file is not a Primed workspace export (version 1).")
    for key in ("workspace_id", "analyses", "references", "clips"):
        if key not in payload:
            raise ValueError("This export is missing workspace data.")
    if not isinstance(payload["analyses"], dict) or not isinstance(payload["clips"], dict):
        raise ValueError("This export is missing workspace data.")
    if not isinstance(payload["references"], list):
        raise ValueError("This export is missing reference versions.")
    timezone_name = payload.get("timezone") or "UTC"
    _check_timezone(timezone_name)
    analyses = {}
    known_reps: dict[str, dict] = {}
    for analysis_id, analysis in payload["analyses"].items():
        if not isinstance(analysis, dict) or analysis.get("id") != analysis_id:
            raise ValueError("An analysis id does not match its record.")
        profile_id = analysis.get("profile_id")
        if profile_id:
            try:
                profile = get_profile(profile_id)
            except KeyError as exc:
                raise ValueError(f"Unknown movement '{profile_id}' in the export.") from exc
            for rep in analysis.get("reps", []):
                if not isinstance(rep, dict) or not rep.get("id"):
                    raise ValueError("A rep is missing its id.")
                if rep["id"] in known_reps:
                    raise ValueError(f"Duplicate rep id {rep['id']}.")
                known_reps[rep["id"]] = rep
                for metric_id, metric in (rep.get("metrics") or {}).items():
                    spec = profile.metric(metric_id)
                    if spec is None:
                        continue
                    if metric.get("unit") not in (None, spec.unit):
                        raise ValueError(f"{metric_id} uses {metric.get('unit')} instead of {spec.unit}.")
        analysis = _copy(analysis)
        analysis["landmarks"] = analysis.get("landmarks")
        analyses[analysis_id] = analysis
    for ref in payload["references"]:
        if not isinstance(ref, dict) or not ref.get("id"):
            raise ValueError("A reference is missing its id.")
        for rep_id in ref.get("source_rep_ids", []):
            if rep_id not in known_reps:
                raise ValueError(f"Reference {ref['id']} cites missing rep {rep_id}.")
        try:
            get_profile(ref["profile_id"])
        except KeyError as exc:
            raise ValueError(f"Unknown movement '{ref.get('profile_id')}' on a reference.") from exc
    clips = {}
    for clip_id, clip in payload["clips"].items():
        if not isinstance(clip, dict):
            raise ValueError("A clip record is invalid.")
        copied = _copy(clip)
        copied["filename"] = _display_name(str(copied.get("filename") or "clip"))
        copied["media_present"] = False
        for banned in ("path", "playback_path", "overlay_path", "original_path"):
            copied.pop(banned, None)
        clips[clip_id] = copied
    workspace = {
        "export_version": EXPORT_VERSION,
        "workspace_id": str(payload["workspace_id"]),
        "name": str(payload.get("name") or "Imported workspace"),
        "athlete": str(payload.get("athlete") or "Athlete"),
        "timezone": timezone_name,
        "created_at": payload.get("created_at") or _now(),
        "clips": clips,
        "analyses": analyses,
        "references": [_copy(ref) for ref in payload["references"]],
        "synthetic": bool(payload.get("synthetic")),
        "imported_at": _now(),
    }
    return workspace


def merge_workspace(destination: dict, incoming: dict) -> dict:
    """Copy records whose ids are new. Existing ids are left as they are."""
    for clip_id, clip in incoming.get("clips", {}).items():
        destination.setdefault("clips", {}).setdefault(clip_id, clip)
    for analysis_id, analysis in incoming.get("analyses", {}).items():
        destination.setdefault("analyses", {}).setdefault(analysis_id, analysis)
    known = {ref["id"] for ref in destination.get("references", [])}
    for ref in incoming.get("references", []):
        if ref["id"] not in known:
            destination.setdefault("references", []).append(ref)
            known.add(ref["id"])
    return destination


def reattach_matches(clip: dict, sha256: str) -> bool:
    return bool(clip.get("sha256")) and clip.get("sha256") == sha256


def _copy(value):
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value


def _display_name(filename: str) -> str:
    name = filename.replace("\\", "/").split("/")[-1]
    name = name.replace("\x00", "")
    return name[:180] or "clip"


def _check_timezone(name: str) -> None:
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(name)
    except Exception as exc:
        raise ValueError(f"Unknown timezone '{name}'.") from exc
