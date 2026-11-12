"""Streamlit screens for Primed."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from primed.analyze import analyze_clip, ingest_upload, mark_failed
from primed.ask import ask_gemini, build_bundle
from primed.baseline import (
    active_reference,
    approve_reference,
    build_trend,
    compare_rep_traces,
    references_for,
    select_reference,
    set_baseline,
)
from primed.fixture import build_synthetic_workspace
from primed.measure import add_manual_rep, apply_rep_edit, delete_rep, resegment, value_at_time
from primed.pose_draw import draw_pose, draw_reference_shape, normalise_landmarks
from primed.profiles import get_profile, list_profiles, profiles_by_section
from primed.repository import open_repository, session_dir
from primed.ui.styles import css
from primed.video_io import VideoError, extract_clip
from primed.workspace import export_workspace, import_workspace, merge_workspace, reattach_matches


def main() -> None:
    st.markdown(css(), unsafe_allow_html=True)
    _init()
    _header()
    workspace = st.session_state.workspace
    synthetic = workspace.get("synthetic") or any(
        item.get("synthetic") for item in workspace.get("analyses", {}).values()
    )
    if synthetic:
        st.markdown(
            '<div class="synthetic">Synthetic fixture. These figures were calculated from constructed signals, not from a recording.</div>',
            unsafe_allow_html=True,
        )
    _flash()
    view = st.session_state.view
    if view == "sessions":
        _sessions()
    elif view == "review":
        _review()
    elif view == "progress":
        _progress()
    else:
        _playground()


def _init() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.repo = open_repository(session_dir(st.session_state.session_id))
        st.session_state.workspace = st.session_state.repo.load()
    st.session_state.setdefault("view", "playground")
    st.session_state.setdefault("profile_id", "cricket_bowling")
    st.session_state.setdefault("analysis_id", "")
    st.session_state.setdefault("rep_id", "")
    st.session_state.setdefault("pose_overlay", True)
    st.session_state.setdefault("ask", None)


def _commit() -> None:
    st.session_state.repo.save(st.session_state.workspace)


def _flash(level: str | None = None, text: str | None = None) -> None:
    if level and text:
        st.session_state["_flash"] = (level, text)
        return
    item = st.session_state.pop("_flash", None)
    if not item:
        return
    getattr(st, item[0])(item[1])


def _go(view: str) -> None:
    st.session_state.view = view


def _set_state(key: str, value) -> None:
    st.session_state[key] = value


def _header() -> None:
    brand, play, sessions, progress, upload = st.columns([1.3, 1, 1, 1, 1.15])
    with brand:
        st.markdown('<p class="wordmark">Primed</p>', unsafe_allow_html=True)
    view = st.session_state.view
    with play:
        st.button("Playground", width="stretch", type="primary" if view == "playground" else "secondary", on_click=_go, args=("playground",))
    with sessions:
        st.button("Sessions", width="stretch", type="primary" if view == "sessions" else "secondary", on_click=_go, args=("sessions",))
    with progress:
        st.button("Progress", width="stretch", type="primary" if view == "progress" else "secondary", on_click=_go, args=("progress",))
    with upload:
        st.button("Upload clip", width="stretch", type="primary", on_click=_go, args=("playground",))


def _workspace():
    return st.session_state.workspace


def _profile():
    return get_profile(st.session_state.profile_id)


def _select_profile(profile_id: str) -> None:
    st.session_state.profile_id = profile_id
    st.session_state.analysis_id = ""
    st.session_state.rep_id = ""


def _analyses_for(profile_id: str) -> list[dict]:
    rows = [item for item in _workspace().get("analyses", {}).values() if item.get("profile_id") == profile_id]
    rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return rows


def _current_analysis() -> dict | None:
    workspace = _workspace()
    selected = st.session_state.get("analysis_id")
    if selected and selected in workspace.get("analyses", {}):
        analysis = workspace["analyses"][selected]
        if analysis.get("profile_id") == st.session_state.profile_id:
            return analysis
    matches = _analyses_for(st.session_state.profile_id)
    if not matches:
        return None
    st.session_state.analysis_id = matches[0]["id"]
    return matches[0]


def _current_rep(analysis: dict | None) -> dict | None:
    if not analysis or not analysis.get("reps"):
        return None
    selected = st.session_state.get("rep_id")
    for rep in analysis["reps"]:
        if rep["id"] == selected:
            return rep
    st.session_state.rep_id = analysis["reps"][0]["id"]
    return analysis["reps"][0]


def _fmt(value, unit: str) -> str:
    if value is None:
        return "Unavailable"
    if unit == "deg":
        return f"{float(value):.1f}°"
    if unit == "s":
        return f"{float(value):.2f} s"
    if unit == "σ":
        return f"{float(value):.2f}σ"
    if unit == "rpm":
        return f"{float(value):.0f} /min"
    return f"{float(value):.2f}"


def _library() -> None:
    st.markdown("**Movement library**")
    st.caption(f"{len(list_profiles())} movements")
    query = st.text_input("Search movements", placeholder="Search movements", label_visibility="collapsed")
    needle = query.strip().lower()
    for section, profiles in profiles_by_section():
        visible = [profile for profile in profiles if needle in profile.name.lower() or needle in section.lower()]
        if not visible:
            continue
        st.markdown(f"**{section}**")
        for profile in visible:
            selected = profile.id == st.session_state.profile_id
            st.button(
                profile.name,
                key=f"lib_{profile.id}",
                width="stretch",
                type="primary" if selected else "secondary",
                on_click=_select_profile,
                args=(profile.id,),
            )


def _playground() -> None:
    st.markdown("# Movement playground")
    st.caption("Pick an activity, review the footage, and set the movement you want to repeat.")
    profile = _profile()
    library, stage, baseline = st.columns([1.05, 2.35, 1.15], gap="medium")
    with library:
        _library()
    with stage:
        _stage(profile)
    with baseline:
        _baseline_panel(profile)


def _stage(profile) -> None:
    analysis = _current_analysis()
    eyebrow = f"{profile.name} · {profile.view_label(analysis['view']) if analysis else profile.view_label(profile.default_view)}"
    st.markdown(f'<p class="eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.markdown(f"# {profile.name}")
    st.write(profile.summary)
    with st.expander("What this movement can measure"):
        for item in profile.available:
            st.write(f"Available: {item}")
        for item in profile.unavailable:
            st.write(f"Unavailable: {item}")
        st.caption(profile.segmentation_note)
    matches = _analyses_for(profile.id)
    if matches:
        labels = {item["id"]: item.get("filename") or item["id"] for item in matches}
        choice = st.selectbox(
            "Clip",
            list(labels),
            format_func=lambda item: labels[item],
            index=list(labels).index(analysis["id"]) if analysis else 0,
            key=f"clip_choice_{profile.id}",
        )
        if choice != st.session_state.analysis_id:
            st.session_state.analysis_id = choice
            st.session_state.rep_id = ""
            analysis = _workspace()["analyses"][choice]
    else:
        analysis = None
    pending = _upload_box(profile)
    if analysis is None:
        if not pending:
            st.info("Upload a clip to measure this movement. No example recording is included in this build.")
        if st.button(
            "Preview charts with a labelled synthetic fixture",
            key="fixture_from_playground",
            help="Replaces this session with labelled chart data. It is not a recording.",
        ):
            _load_fixture()
        return
    _player(analysis, profile)
    rep = _current_rep(analysis)
    if rep:
        if st.button("Open this rep in review", key="open_review"):
            st.session_state.view = "review"
            st.rerun()


def _upload_box(profile) -> bool:
    st.markdown("**Upload clip**")
    uploaded = st.file_uploader(
        "Clip file",
        type=["mp4", "mov", "m4v", "webm"],
        key=f"upload_{profile.id}",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MB")
    variant = st.selectbox(
        "Variant",
        [item[0] for item in profile.variants],
        format_func=lambda item: profile.variant_label(item),
        key=f"variant_{profile.id}",
    )
    view = st.selectbox(
        "Camera view",
        [item[0] for item in profile.views],
        format_func=lambda item: profile.view_label(item),
        key=f"view_{profile.id}",
    )
    facing = ""
    if view == "side":
        facing = st.selectbox("Facing", ["right", "left"], format_func=lambda item: f"Facing {item} of frame", key=f"facing_{profile.id}")
    handedness = ""
    if profile.handedness_required:
        handedness = st.selectbox("Handedness", ["right", "left"], format_func=lambda item: item.capitalize(), key=f"hand_{profile.id}")
    context = {}
    for field, label, kind in profile.context_fields:
        if kind == "number":
            value = st.number_input(label, min_value=0.0, value=0.0, step=1.0, key=f"ctx_{profile.id}_{field}")
            if value:
                context[field] = value
        elif kind == "side":
            context[field] = st.selectbox(label, ["", "left", "right"], key=f"ctx_{profile.id}_{field}")
            if not context[field]:
                context.pop(field)
        else:
            text = st.text_input(label, key=f"ctx_{profile.id}_{field}")
            if text.strip():
                context[field] = text.strip()
    recorded = st.date_input("Session date", key=f"date_{profile.id}")
    st.button("Open example", disabled=True, help="No example recording is included in this build.", key=f"example_{profile.id}")
    if st.button("Analyse clip", type="primary", disabled=uploaded is None, key=f"analyse_{profile.id}"):
        _run_upload(uploaded, profile, variant, view, facing, handedness, context, recorded)
    notice = st.session_state.get("upload_notice") or ""
    if notice:
        st.error(notice)
    return uploaded is not None


def _run_upload(uploaded, profile, variant, view, facing, handedness, context, recorded) -> None:
    recorded_at = datetime(recorded.year, recorded.month, recorded.day, 12, 0, tzinfo=timezone.utc).isoformat()
    try:
        with st.status(f"Analysing {uploaded.name}", expanded=True) as status:
            clip = ingest_upload(
                st.session_state.repo,
                _workspace(),
                uploaded.getvalue(),
                uploaded.name,
                {
                    "profile_id": profile.id,
                    "variant": variant,
                    "view": view,
                    "facing": facing,
                    "handedness": handedness,
                    "context": context,
                    "athlete": _workspace().get("athlete") or "Athlete",
                    "recorded_at": recorded_at,
                },
            )
            status.write(f"{uploaded.name} passed the file checks.")

            def _progress(fraction: float, message: str) -> None:
                status.write(message)

            analysis = analyze_clip(st.session_state.repo, _workspace(), clip["id"], progress=_progress)
            status.update(label=f"{uploaded.name} is ready", state="complete")
        _commit()
        st.session_state.analysis_id = analysis["id"]
        st.session_state.rep_id = analysis["reps"][0]["id"] if analysis.get("reps") else ""
        st.session_state["upload_notice"] = ""
        _flash("success", f"{uploaded.name} analysed. {len(analysis.get('reps') or [])} rep(s) proposed.")
    except (VideoError, OSError, ValueError) as exc:
        mark_failed(_workspace(), None, str(exc))
        _commit()
        st.session_state["upload_notice"] = f"{uploaded.name}: {exc}"
    st.rerun()


def _player(analysis: dict, profile) -> None:
    overlay = st.toggle("Pose overlay", value=True, key=f"overlay_{analysis['id']}")
    frames = analysis.get("landmarks") or []
    clip = _workspace().get("clips", {}).get(analysis.get("clip_id") or "", {})
    playback = _media(analysis, "overlay.mp4" if overlay else "playback.mp4")
    if playback is None:
        playback = _media(analysis, "playback.mp4")
    if playback is not None:
        st.video(str(playback))
        st.caption("This player seeks in whole seconds. The frame preview below is the exact analysed frame.")
    elif not frames:
        st.info("The trace is saved. Reattach the original footage from Sessions to see frames.")
    if frames:
        key = f"frame_{analysis['id']}"
        if key not in st.session_state or st.session_state[key] >= len(frames):
            st.session_state[key] = 0
        index = st.slider("Frame preview", 0, len(frames) - 1, key=key)
        frame = frames[int(index)]
        _show_still(analysis, frame, overlay, None)
        moment = float(frame["t"])
        phase = _phase_at(analysis, moment)
        st.caption(
            f"Exact analysed frame at {moment:.2f} s · source frame {frame['index']}"
            + (f" · {phase}" if phase else "")
            + f" · timing {analysis.get('timing_mode', 'unknown')}"
        )
        _readouts(profile, analysis, moment)
    rep = _current_rep(analysis)
    if analysis.get("reps"):
        choice = st.selectbox(
            "Rep",
            [item["id"] for item in analysis["reps"]],
            format_func=lambda item: _rep_label(analysis, item),
            key=f"rep_select_{analysis['id']}",
        )
        st.session_state.rep_id = choice
        rep = next(item for item in analysis["reps"] if item["id"] == choice)
        st.caption(_rep_summary(rep))
    else:
        st.warning("No rep was detected. You can mark one manually in review. Nothing has been estimated.")
    for warning in analysis.get("warnings") or []:
        st.caption(warning)
    if clip.get("error"):
        st.error(clip["error"])


def _readouts(profile, analysis: dict, moment: float) -> None:
    cells = []
    for series_name, label, unit in profile.frame_readouts:
        value = value_at_time(analysis, series_name, moment)
        note = ""
        if series_name == "trunk_tilt":
            facing = analysis.get("facing") or "unspecified"
            note = f"Positive is forward lean · facing {facing}" if value is not None else "Needs a side view with facing set"
        elif series_name in analysis.get("sides", {}):
            note = ""
        elif series_name in ("knee", "elbow", "hip") and analysis.get("sides", {}).get(series_name):
            note = f"{analysis['sides'][series_name].capitalize()} side"
        cells.append(f"<div><span>{label}</span><strong>{_fmt(value, unit)}</strong><em>{note}</em></div>")
    st.markdown(f'<div class="readouts">{"".join(cells)}</div>', unsafe_allow_html=True)
    st.caption("2D projected measurements. A missing joint is shown as unavailable.")


def _phase_at(analysis: dict, moment: float) -> str:
    for rep in analysis.get("reps") or []:
        if rep["start_s"] <= moment <= rep["end_s"]:
            for phase in rep.get("phases") or []:
                if phase["start_s"] <= moment <= phase["end_s"] + 1e-6:
                    return phase["label"]
            return f"Rep {rep['index']}"
    return ""


def _rep_label(analysis: dict, rep_id: str) -> str:
    rep = next(item for item in analysis["reps"] if item["id"] == rep_id)
    quality = rep["quality"].replace("_", " ")
    label = rep.get("set_label") or "no set"
    return f"Rep {rep['index']} · {rep['start_s']:.2f}–{rep['end_s']:.2f} s · {quality} · {label}"


def _rep_summary(rep: dict) -> str:
    parts = [f"Quality: {rep['quality'].replace('_', ' ')}"]
    for metric in rep.get("metrics", {}).values():
        pass
    shown = []
    for key, metric in list(rep.get("metrics", {}).items())[:4]:
        text = _fmt(metric.get("value"), metric.get("unit") or "")
        if metric.get("status") == "candidate":
            text += " candidate"
        elif not metric.get("valid"):
            text = "Unavailable"
        shown.append(f"{key.replace('_', ' ')} {text}")
    if shown:
        parts.append(" · ".join(shown))
    return " · ".join(parts)


def _media(analysis: dict, name: str):
    clip_id = analysis.get("clip_id")
    if not clip_id:
        return None
    path = st.session_state.repo.clip_dir(clip_id) / name
    if path.is_file():
        return path
    return None


def _show_still(analysis: dict, frame: dict, overlay: bool, reference_points: dict | None) -> None:
    path = None
    if analysis.get("clip_id"):
        candidate = st.session_state.repo.clip_dir(analysis["clip_id"]) / "frames" / f"{int(frame['index']):06d}.jpg"
        if candidate.is_file():
            path = candidate
    if path is None:
        st.caption("No exact frame is stored for this session. The trace is still available.")
        return
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        st.caption("The saved frame could not be read.")
        return
    if overlay:
        image = draw_pose(image, frame.get("lm"))
        if reference_points:
            image = draw_reference_shape(image, frame.get("lm"), reference_points)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")


def _baseline_panel(profile) -> None:
    analysis = _current_analysis()
    variant = analysis["variant"] if analysis else profile.default_variant
    view = analysis["view"] if analysis else profile.default_view
    reference = active_reference(_workspace(), profile.id, variant, view)
    st.markdown(f"**{profile.name} baseline**")
    with st.container(border=True):
        st.markdown('<span class="mint-marker"></span>', unsafe_allow_html=True)
        if reference is None:
            st.markdown("**No baseline yet**")
            st.caption("Reference data still being collected")
            other = [item for item in _workspace().get("references", []) if item.get("active")]
            if other:
                other_profile = get_profile(other[0]["profile_id"])
                st.caption(f"The active baseline is for {other_profile.name} and is not used for {profile.name}.")
        else:
            st.markdown(f"**Version {reference['version']}**")
            st.caption(f"{reference['provenance'].replace('-', ' ').capitalize()} · {reference['created_at'][:10]}")
            st.caption(f"Active for {profile.variant_label(reference['variant'])} · {profile.view_label(reference['view'])}")
            headline = next(iter(reference["scalars"].values()), None)
            if headline and headline.get("value") is not None:
                spec = profile.metrics[0]
                st.markdown(f"**{_fmt(reference['scalars'][spec.id]['value'], spec.unit)}**")
                st.caption(f"{spec.label} · {headline['n']} reference rep(s)")
            if reference.get("synthetic"):
                st.caption("Synthetic reference. Not a filmed baseline.")
    st.markdown("**Set a new reference**")
    if analysis and analysis.get("reps"):
        rep_ids = [rep["id"] for rep in analysis["reps"]]
        primary = st.selectbox("Candidate rep", rep_ids, format_func=lambda item: _rep_label(analysis, item), key=f"base_rep_{analysis['id']}")
        extras = st.multiselect(
            "Other approved reps",
            [item for item in rep_ids if item != primary],
            format_func=lambda item: _rep_label(analysis, item),
            key=f"base_extra_{analysis['id']}",
        )
        allow_weak = st.checkbox("Include an incomplete rep anyway", key=f"weak_{analysis['id']}")
        if st.button("Set as baseline", type="primary", key=f"set_base_{analysis['id']}"):
            try:
                created = set_baseline(
                    _workspace(),
                    [primary, *extras],
                    provenance="self-selected",
                    allow_weak=allow_weak,
                )
                _commit()
                _flash("success", f"Saved {profile.name} baseline version {created['version']}.")
            except ValueError as exc:
                _flash("error", str(exc))
            st.rerun()
    else:
        st.caption("Analyse a clip, then choose the rep you want to repeat.")
    history = references_for(_workspace(), profile.id, variant, view)
    if history:
        st.markdown("**Baseline history**")
        for item in reversed(history):
            label = f"Version {item['version']} · {item['provenance']} · {item['created_at'][:10]}"
            cols = st.columns([2.2, 1])
            cols[0].caption(("Active · " if item.get("active") else "") + label)
            if cols[1].button("Use", key=f"use_{item['id']}"):
                select_reference(_workspace(), item["id"])
                _commit()
                _flash("success", f"Version {item['version']} is now the active baseline. Its stored measurements were not changed.")
                st.rerun()
        if reference and st.button("Approve reference", key=f"approve_{reference['id']}"):
            approve_reference(_workspace(), reference["id"], "coach-approved")
            _commit()
            _flash("success", "Recorded a coach approval on this reference. The reference id is unchanged.")
            st.rerun()


def _load_fixture() -> None:
    st.session_state.workspace = build_synthetic_workspace()
    _commit()
    st.session_state.profile_id = "squat"
    st.session_state.view = "progress"
    st.session_state.analysis_id = ""
    st.session_state.rep_id = ""
    st.rerun()


def _sessions() -> None:
    st.markdown("# Sessions")
    st.write("Clips and measurements in this browser session. Export the workspace to continue on another day. The file does not contain the video.")
    workspace = _workspace()
    name = st.text_input("Workspace", value=workspace.get("name") or "My workspace")
    athlete = st.text_input("Athlete", value=workspace.get("athlete") or "Athlete")
    timezone_name = st.text_input("Timezone", value=workspace.get("timezone") or "UTC", help="Days on the progress page use this timezone.")
    if st.button("Save workspace details"):
        workspace["name"] = name.strip() or "My workspace"
        workspace["athlete"] = athlete.strip() or "Athlete"
        workspace["timezone"] = timezone_name.strip() or "UTC"
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(workspace["timezone"])
        except Exception:
            st.error("Enter a timezone name such as America/Los_Angeles or Asia/Kolkata.")
            return
        for analysis in workspace.get("analyses", {}).values():
            if not analysis.get("synthetic"):
                analysis["athlete"] = workspace["athlete"]
        _commit()
        st.success("Saved.")
    rows = list(workspace.get("analyses", {}).values())
    rows.sort(key=lambda item: item.get("recorded_at") or "", reverse=True)
    if not rows:
        st.info("No sessions yet.")
    else:
        for analysis in rows:
            profile = get_profile(analysis["profile_id"])
            clip = workspace.get("clips", {}).get(analysis.get("clip_id") or "", {})
            media = "Footage attached" if clip.get("media_present") else "Footage not attached"
            label = f"{profile.name} · {analysis.get('filename') or analysis['id']} · {len(analysis.get('reps') or [])} reps · {media}"
            if analysis.get("synthetic"):
                label = "Synthetic · " + label
            left, right = st.columns([4, 1])
            left.write(label)
            if right.button("Open", key=f"open_{analysis['id']}"):
                st.session_state.profile_id = analysis["profile_id"]
                st.session_state.analysis_id = analysis["id"]
                st.session_state.rep_id = analysis["reps"][0]["id"] if analysis.get("reps") else ""
                st.session_state.view = "review"
                st.rerun()
    st.markdown("### Export and import")
    ids = [item["id"] for item in rows]
    chosen = st.multiselect(
        "Include",
        ids,
        default=ids,
        format_func=lambda item: workspace["analyses"][item].get("filename") or item,
    )
    include_pose = st.checkbox("Include pose samples", value=True)
    if chosen:
        payload = export_workspace(workspace, analysis_ids=chosen, include_pose=include_pose)
        st.download_button(
            "Export workspace",
            data=json.dumps(payload, indent=2),
            file_name="primed-workspace.json",
            mime="application/json",
            type="primary",
        )
        st.caption("The export keeps derived measurements, reference versions, clip ids and hashes. It leaves the video out.")
    uploaded = st.file_uploader("Import workspace JSON", type=["json"], key="import_json")
    mode = st.radio("Import mode", ["Replace this session", "Merge into this session"], horizontal=True)
    if st.button("Import workspace", disabled=uploaded is None):
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))
            imported = import_workspace(payload)
        except (ValueError, json.JSONDecodeError, UnicodeError) as exc:
            st.error(str(exc))
        else:
            if mode.startswith("Replace"):
                st.session_state.workspace = imported
            else:
                merge_workspace(workspace, imported)
            _commit()
            st.success("Imported. Reattach a clip when you want to see its frames again.")
            st.rerun()
    st.markdown("### Reattach footage")
    clip_ids = list(workspace.get("clips", {}))
    if clip_ids:
        clip_id = st.selectbox("Clip", clip_ids, format_func=lambda item: workspace["clips"][item].get("filename") or item)
        clip = workspace["clips"][clip_id]
        st.caption(f"Stored hash {clip.get('sha256', 'unknown')[:12]}… · media {'attached' if clip.get('media_present') else 'missing'}")
        media = st.file_uploader("Original file", type=["mp4", "mov", "m4v", "webm"], key="reattach")
        if st.button("Reattach", disabled=media is None):
            import hashlib

            digest = hashlib.sha256(media.getvalue()).hexdigest()
            if not reattach_matches(clip, digest):
                st.error("This file does not match the stored clip hash.")
            else:
                try:
                    _reattach(clip, media.getvalue(), media.name)
                except VideoError as exc:
                    st.error(str(exc))
                else:
                    st.success("Footage reattached. Open the session and analyse again to rebuild the frames.")
                    st.rerun()
    if st.button(
        "Load labelled synthetic fixture",
        help="Replaces this session with labelled chart data. It is not a recording.",
    ):
        _load_fixture()


def _reattach(clip: dict, file_bytes: bytes, filename: str) -> None:
    from primed.config import limits_from_env
    from primed.video_io import normalize, presentation_times, validate_file

    folder = st.session_state.repo.clip_dir(clip["id"])
    original = folder / "original.mp4"
    original.write_bytes(file_bytes)
    limits = limits_from_env()
    info = validate_file(original, filename, limits)
    normalize(original, folder / "playback.mp4", limits.max_width, info.rotation_deg)
    times, mode = presentation_times(folder / "playback.mp4")
    (folder / "times.json").write_text(json.dumps({"mode": mode, "times": times}))
    clip["media_present"] = True
    clip["timing_mode"] = mode
    clip["filename"] = filename
    _commit()


def _review() -> None:
    profile = _profile()
    analysis = _current_analysis()
    st.markdown(f"# {profile.name}")
    if analysis is None:
        st.info("Choose a movement and analyse a clip in the playground.")
        _library()
        return
    if analysis.get("synthetic"):
        st.caption("Synthetic comparison. The lines come from constructed signals, not a filmed rep.")
    rep = _current_rep(analysis)
    reference = active_reference(_workspace(), analysis["profile_id"], analysis["variant"], analysis["view"])
    if reference is None:
        active_other = [item for item in _workspace().get("references", []) if item.get("active")]
        if active_other:
            st.caption(f"A baseline exists for {get_profile(active_other[0]['profile_id']).name}. It is not applied to {profile.name}.")
    left, right = st.columns([1.15, 1.15], gap="medium")
    with left:
        st.markdown("**Selected rep**")
        if analysis.get("reps"):
            choice = st.selectbox(
                "Rep",
                [item["id"] for item in analysis["reps"]],
                index=[item["id"] for item in analysis["reps"]].index(rep["id"]) if rep else 0,
                format_func=lambda item: _rep_label(analysis, item),
                key=f"review_rep_{analysis['id']}",
            )
            st.session_state.rep_id = choice
            rep = next(item for item in analysis["reps"] if item["id"] == choice)
        overlay = st.toggle("Pose overlay", value=True, key=f"review_overlay_{analysis['id']}")
        _review_media(analysis, rep, reference, overlay)
    with right:
        st.markdown("**Measured traces**")
        if rep:
            _trace_charts(analysis, rep, reference)
        else:
            st.info("Mark a rep to calculate measurements.")
    if rep:
        _rep_table(analysis)
        _editor(analysis, rep, profile)
        _evidence(analysis, rep)
    _ask_panel(profile, analysis, rep)


def _review_media(analysis, rep, reference, overlay: bool) -> None:
    frames = analysis.get("landmarks") or []
    playback = _media(analysis, "overlay.mp4" if overlay else "playback.mp4") or _media(analysis, "playback.mp4")
    if playback is not None:
        st.video(str(playback))
        st.caption("This player seeks in whole seconds. It is not a frame-accurate control.")
    if not frames or rep is None:
        if analysis.get("synthetic"):
            st.caption("This synthetic session has traces and no footage.")
        return
    progress_key = f"progress_{analysis['id']}_{rep['id']}"
    progress = st.slider("Phase progress", 0.0, 1.0, key=progress_key)
    moment = _time_at_progress(rep.get("phases") or [], progress)
    if moment is None:
        moment = rep["start_s"]
    frame = min(frames, key=lambda item: abs(float(item["t"]) - moment))
    points = _reference_shape(reference, progress) if overlay else None
    _show_still(analysis, frame, overlay, points)
    phase = _phase_at(analysis, float(frame["t"]))
    st.caption(
        f"Exact analysed frame at {float(frame['t']):.2f} s · source frame {frame['index']}"
        + (f" · {phase}" if phase else "")
    )
    if points:
        st.caption("The green outline is the reference rep at the same phase progress. It is a shape guide, not a shared clock.")
    for phase_item in rep.get("phases") or []:
        st.button(
            phase_item["label"],
            key=f"jump_{rep['id']}_{phase_item['id']}_{phase_item['start_s']}",
            on_click=_set_state,
            args=(progress_key, _progress_for_time(rep.get("phases") or [], phase_item["start_s"])),
        )


def _time_at_progress(phases: list[dict], progress: float) -> float | None:
    usable = [phase for phase in phases if phase["end_s"] >= phase["start_s"]]
    if not usable:
        return None
    position = min(0.999, max(0.0, progress)) * len(usable)
    index = min(len(usable) - 1, int(position))
    fraction = position - index
    phase = usable[index]
    return float(phase["start_s"]) + fraction * (float(phase["end_s"]) - float(phase["start_s"]))


def _progress_for_time(phases: list[dict], moment: float) -> float:
    usable = [phase for phase in phases if phase["end_s"] >= phase["start_s"]]
    if not usable:
        return 0.0
    for index, phase in enumerate(usable):
        if phase["start_s"] <= moment <= phase["end_s"] + 1e-6:
            span = max(1e-6, phase["end_s"] - phase["start_s"])
            fraction = min(1.0, max(0.0, (moment - phase["start_s"]) / span))
            return (index + fraction) / len(usable)
    return 0.0


def _reference_shape(reference: dict | None, progress: float) -> dict | None:
    if not reference:
        return None
    preview = reference.get("preview_rep_id")
    for analysis in _workspace().get("analyses", {}).values():
        for rep in analysis.get("reps", []):
            if rep["id"] != preview:
                continue
            frames = analysis.get("landmarks") or []
            if not frames:
                return None
            moment = _time_at_progress(rep.get("phases") or [], progress) or rep["start_s"]
            frame = min(frames, key=lambda item: abs(float(item["t"]) - moment))
            normalised = normalise_landmarks(frame.get("lm"), int(analysis.get("width") or 1), int(analysis.get("height") or 1))
            if not normalised:
                return None
            return normalised["points"]
    return None


def _trace_charts(analysis: dict, rep: dict, reference: dict | None) -> None:
    charts = compare_rep_traces(analysis, rep, reference)
    if not charts:
        st.caption("No trace is available for this movement.")
        return
    for chart in charts[:4]:
        figure = go.Figure()
        current = chart["current"]
        figure.add_trace(
            go.Scatter(
                x=current["x"],
                y=current["y"],
                mode="lines",
                name="Selected rep",
                line={"color": "#1243FF", "width": 2},
                connectgaps=False,
                hovertemplate="Phase progress %{x:.2f}<br>%{y:.2f}<extra></extra>",
            )
        )
        reference_trace = chart.get("reference") or {}
        if reference_trace.get("y"):
            figure.add_trace(
                go.Scatter(
                    x=reference_trace.get("x") or current["x"],
                    y=reference_trace["y"],
                    mode="lines",
                    name="Reference",
                    line={"color": "#12805C", "width": 2, "dash": "dash"},
                    connectgaps=False,
                )
            )
        title = chart["label"]
        if analysis.get("synthetic"):
            title += " · synthetic"
        _quiet(figure, title, "Phase progress", chart["unit"])
        st.plotly_chart(figure, width="stretch", theme=None, key=f"trace_{analysis['id']}_{rep['id']}_{chart['series']}")
    st.caption("Solid is the selected rep and dashed is the reference mean. Phase progress aligns shape only. Timing stays on the original timestamps. Gaps are missing samples.")
    _time_chart(analysis, rep, charts[0])


def _time_chart(analysis: dict, rep: dict, chart: dict) -> None:
    series_name = chart["series"]
    plotted = (analysis.get("plot_series") or {}).get(series_name) or {}
    values = plotted.get("y") or analysis.get("series", {}).get(series_name) or []
    xs, ys = [], []
    for moment, value in zip(analysis["times"], values):
        if rep["start_s"] - 0.02 <= float(moment) <= rep["end_s"] + 0.02:
            xs.append(float(moment))
            ys.append(value)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Selected rep", line={"color": "#0B8F78", "width": 2}, connectgaps=False))
    _quiet(figure, f"{chart['label']} by time", "Seconds", chart["unit"])
    st.plotly_chart(figure, width="stretch", theme=None, key=f"time_{analysis['id']}_{rep['id']}_{series_name}")
    st.caption("Seconds are the clip timestamps. A gap longer than 0.2 s is left empty. Shorter gaps may be bridged for the line only.")


def _quiet(figure: go.Figure, title: str, x_title: str, y_title: str) -> None:
    figure.update_layout(
        title={"text": title, "font": {"family": "Bricolage Grotesque, sans-serif", "size": 16, "color": "#101831"}},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 48, "r": 12, "t": 42, "b": 36},
        font={"family": "Instrument Sans, sans-serif", "color": "#101831", "size": 12},
        height=240,
        legend={"orientation": "h", "y": 1.12},
    )
    figure.update_xaxes(title=x_title, showgrid=False, zeroline=False, linecolor="#D5DCEC")
    figure.update_yaxes(title=y_title, showgrid=False, zeroline=False, linecolor="#D5DCEC")


def _rep_table(analysis: dict) -> None:
    st.markdown("**Reps**")
    rows = []
    for rep in analysis["reps"]:
        row = {
            "Rep": rep["index"],
            "Id": rep["id"],
            "Start": round(rep["start_s"], 2),
            "End": round(rep["end_s"], 2),
            "Set": rep.get("set_label") or "",
            "Quality": rep["quality"].replace("_", " "),
        }
        for key, metric in rep.get("metrics", {}).items():
            row[key.replace("_", " ")] = _fmt(metric.get("value"), metric.get("unit") or "") if metric.get("valid") or metric.get("status") == "candidate" else "Unavailable"
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)


def _editor(analysis: dict, rep: dict, profile) -> None:
    with st.expander("Correct this rep"):
        start = st.number_input("Start (seconds)", value=float(rep["start_s"]), step=0.05, key=f"edit_start_{rep['id']}")
        end = st.number_input("End (seconds)", value=float(rep["end_s"]), step=0.05, key=f"edit_end_{rep['id']}")
        set_label = st.text_input("Set label", value=rep.get("set_label") or "", key=f"edit_set_{rep['id']}", help="Sets on the progress page use this label.")
        event_times = {}
        confirm = set()
        for event in rep.get("events") or []:
            cols = st.columns([2, 1])
            current = float(event["t"]) if event.get("t") is not None else float(rep["start_s"])
            event_times[event["id"]] = cols[0].number_input(
                f"{event['label']} ({event['status']})",
                value=current,
                step=0.05,
                key=f"ev_{rep['id']}_{event['id']}",
            )
            if cols[1].checkbox("Confirm", value=event["status"] in ("confirmed", "measured"), key=f"cf_{rep['id']}_{event['id']}"):
                confirm.add(event["id"])
            if event.get("note"):
                st.caption(event["note"])
        if st.button("Save corrections", type="primary", key=f"save_{rep['id']}"):
            try:
                apply_rep_edit(
                    analysis,
                    rep["id"],
                    start_s=float(start),
                    end_s=float(end),
                    set_label=set_label,
                    event_times={key: float(value) for key, value in event_times.items()},
                    confirm_ids=confirm,
                )
                _commit()
                st.success("Saved the corrected boundaries with the source timestamps.")
            except ValueError as exc:
                st.error(str(exc))
        cols = st.columns(3)
        if cols[0].button("Mark whole clip as one rep", key=f"manual_{analysis['id']}"):
            add_manual_rep(analysis)
            _commit()
            st.rerun()
        if cols[1].button("Re-run automatic segmentation", key=f"reseg_{analysis['id']}"):
            resegment(analysis)
            _commit()
            st.success("Automatic reps were replaced. Manually edited reps were kept.")
            st.rerun()
        if cols[2].button("Delete this rep", key=f"del_{rep['id']}"):
            delete_rep(analysis, rep["id"])
            st.session_state.rep_id = ""
            _commit()
            st.rerun()


def _evidence(analysis: dict, rep: dict) -> None:
    st.markdown("**Evidence**")
    events = [event for event in rep.get("events") or [] if event.get("t") is not None]
    if not events:
        st.caption("Mark an event to pin an exact frame.")
        return
    event = st.selectbox("Event", events, format_func=lambda item: f"{item['label']} · {item['t']:.2f} s · {item['status']}", key=f"evidence_{rep['id']}")
    frames = analysis.get("landmarks") or []
    if frames:
        frame = min(frames, key=lambda item: abs(float(item["t"]) - float(event["t"])))
        _show_still(analysis, frame, True, None)
        st.caption(f"Exact analysed frame at {float(frame['t']):.2f} s for {event['label']}. Source frame {frame['index']}.")
    else:
        st.caption("The event time is saved. Reattach the footage to see the frame.")
    playback = _media(analysis, "playback.mp4")
    if playback is not None and st.button("Extract evidence clip", key=f"extract_{rep['id']}_{event['id']}"):
        dest = st.session_state.repo.clip_dir(analysis["clip_id"]) / f"evidence_{event['id']}.mp4"
        start = max(0.0, float(event["t"]) - 0.4)
        extract_clip(playback, dest, start, 0.8)
        st.session_state[f"evidence_path_{rep['id']}"] = str(dest)
    path = st.session_state.get(f"evidence_path_{rep['id']}")
    if path:
        st.video(path)
        st.caption("Extracted evidence clip. The player is not frame-accurate; use the still above for the exact frame.")


def _progress() -> None:
    st.markdown("# Movement drift")
    workspace = _workspace()
    analyses = list(workspace.get("analyses", {}).values())
    if not analyses:
        st.info("No measured sessions yet. Upload a clip, or preview the charts with a labelled synthetic fixture.")
        if st.button("Preview with a labelled synthetic fixture", type="primary"):
            _load_fixture()
        return
    profile_ids = list(dict.fromkeys(item["profile_id"] for item in analyses))
    if st.session_state.profile_id not in profile_ids:
        st.session_state.profile_id = profile_ids[0]
    profile = get_profile(
        st.selectbox(
            "Movement",
            profile_ids,
            index=profile_ids.index(st.session_state.profile_id),
            format_func=lambda item: get_profile(item).name,
            key="progress_profile",
        )
    )
    st.session_state.profile_id = profile.id
    versions = [item for item in workspace.get("references", []) if item["profile_id"] == profile.id]
    versions.sort(key=lambda item: (item["variant"], item["view"], item["version"]))
    if not versions:
        st.info(f"Set a {profile.name} baseline in the playground before comparing drift. Another movement's baseline is not reused.")
        return
    labels = {
        item["id"]: f"Version {item['version']} · {profile.variant_label(item['variant'])} · {profile.view_label(item['view'])} · {item['provenance']}"
        + (" · active" if item.get("active") else "")
        for item in versions
    }
    active = next((item for item in versions if item.get("active")), versions[-1])
    reference_id = st.selectbox(
        "Reference version",
        list(labels),
        index=list(labels).index(active["id"]),
        format_func=lambda item: labels[item],
        key="progress_ref",
    )
    reference = next(item for item in versions if item["id"] == reference_id)
    athletes = sorted({item.get("athlete") or "Athlete" for item in analyses if item["profile_id"] == profile.id})
    loads = sorted({(item.get("context") or {}).get("load_kg") for item in analyses if (item.get("context") or {}).get("load_kg") not in (None, "")})
    filters = st.columns(4)
    athlete = filters[0].selectbox("Athlete", ["Any", *athletes])
    filters[1].caption(f"Variant {profile.variant_label(reference['variant'])}")
    filters[2].caption(f"View {profile.view_label(reference['view'])}")
    load_choice = filters[3].selectbox("Load", ["Any", *loads], format_func=lambda item: "Any" if item == "Any" else f"{item:g} kg") if loads else "Any"
    mode = st.segmented_control("Grouping", ["Reps", "Sets", "Days"], default="Reps", key="progress_mode")
    scale = st.segmented_control("Scale", ["Raw difference", "Standardised (σ)"], default="Raw difference", key="progress_scale")
    mode_key = {"Reps": "reps", "Sets": "sets", "Days": "days"}[mode or "Reps"]
    standardised = scale == "Standardised (σ)"
    timezone_name = workspace.get("timezone") or "UTC"
    trend = build_trend(
        workspace,
        reference,
        mode=mode_key,
        standardised=standardised,
        athlete=None if athlete == "Any" else athlete,
        load=None if load_choice == "Any" else float(load_choice),
        timezone_name=timezone_name,
    )
    st.caption(trend["grouping"])
    st.caption(
        f"Reference version {reference['version']} · {reference['provenance']} · {trend['scoped_reps']} comparable reps · {trend['group_count']} groups."
    )
    if reference.get("synthetic") or trend.get("synthetic"):
        st.caption("Synthetic comparison. The guides are calculated from the fixture, not from filmed reps.")
    st.caption("2σ and 3σ guides describe the reference spread. They are not a confidence interval, an injury judgement, or coaching advice.")
    _trend_grid(trend, standardised)
    _selection_panel(trend)
    excluded = trend["excluded"]
    st.caption(f"Excluded records: {len(excluded)}")
    if excluded:
        with st.expander("Excluded data"):
            counts: dict[str, int] = {}
            for item in excluded:
                counts[item["reason"]] = counts.get(item["reason"], 0) + 1
            for reason, count in sorted(counts.items()):
                st.write(f"{reason}: {count}")


def _trend_grid(trend: dict, standardised: bool) -> None:
    ymax = 3.4
    if standardised:
        for series in trend["series"]:
            for point in series["points"]:
                if point["y"] is not None:
                    ymax = max(ymax, float(point["y"]) * 1.15)
    columns = st.columns(3)
    for index, series in enumerate(trend["series"]):
        with columns[index % 3]:
            if series["unavailable_reason"]:
                st.markdown(f"**{series['label']}**")
                st.caption(series["unavailable_reason"])
                st.caption(series["coverage"])
                continue
            figure = go.Figure()
            xs = [point["x"] for point in series["points"]]
            ys = [point["y"] for point in series["points"]]
            custom = [",".join(point["rep_ids"]) for point in series["points"]]
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=series["label"],
                    line={"color": "#1243FF", "width": 2},
                    marker={"size": 7},
                    customdata=custom,
                    hovertext=[point["hover"] for point in series["points"]],
                    hovertemplate="%{hovertext}<extra></extra>",
                    connectgaps=False,
                )
            )
            if standardised and series.get("sigma_ok"):
                figure.add_hline(y=2, line_dash="dash", line_color="#C48A12", annotation_text="2σ", annotation_position="right")
                figure.add_hline(y=3, line_dash="dot", line_color="#D4534A", annotation_text="3σ", annotation_position="right")
                figure.update_yaxes(range=[0, ymax])
            else:
                figure.add_hline(y=0, line_color="#C5CEDF", line_width=1)
            subtitle = series["coverage"]
            if series.get("ref_n") is not None and series.get("ref_value") is not None:
                subtitle += f" · reference n={series['ref_n']}"
            _quiet(figure, series["label"], "", "σ" if standardised and series.get("sigma_ok") else series["raw_unit"])
            figure.update_layout(title={"text": f"{series['label']}<br><sup>{subtitle}</sup>"}, showlegend=False, height=230)
            state = st.plotly_chart(
                figure,
                width="stretch",
                theme=None,
                on_select="rerun",
                selection_mode="points",
                key=f"trend_{series['metric_id']}_{trend['mode']}_{int(standardised)}_{trend['reference_id']}",
            )
            picked = _plotly_custom(state)
            if picked:
                st.session_state["trend_pick"] = picked


def _plotly_custom(state) -> str | None:
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if isinstance(custom, (list, tuple)):
        custom = custom[0] if custom else None
    return str(custom) if custom else None


def _selection_panel(trend: dict) -> None:
    st.markdown("**Selected reps**")
    options = []
    for series in trend["series"]:
        for point in series["points"]:
            for rep_id in point["rep_ids"]:
                if rep_id not in options:
                    options.append(rep_id)
    if not options:
        st.caption("No reps match this filter.")
        return
    picked = st.session_state.get("trend_pick")
    default = []
    if picked:
        default = [item for item in picked.split(",") if item in options]
    choice = st.selectbox("Rep", default or options, key="progress_rep_fallback")
    if default:
        st.caption("The chart selection is listed first. The rep menu remains available if the chart does not report a selection.")
        choice = default[0]
    analysis, rep = _find_rep(choice)
    if rep is None:
        return
    profile = get_profile(analysis["profile_id"])
    st.write(f"{profile.name} · {analysis.get('filename') or analysis['id']} · rep {rep['index']} · {rep['start_s']:.2f}–{rep['end_s']:.2f} s")
    bits = []
    for spec in profile.metrics:
        metric = rep["metrics"].get(spec.id)
        if not metric:
            continue
        bits.append(f"{spec.label}: {_fmt(metric.get('value'), spec.unit) if metric.get('valid') else 'Unavailable'}")
    st.caption(" · ".join(bits))
    if st.button("Open rep", type="primary"):
        st.session_state.profile_id = analysis["profile_id"]
        st.session_state.analysis_id = analysis["id"]
        st.session_state.rep_id = rep["id"]
        st.session_state.view = "review"
        st.rerun()


def _find_rep(rep_id: str) -> tuple[dict | None, dict | None]:
    for analysis in _workspace().get("analyses", {}).values():
        for rep in analysis.get("reps", []):
            if rep["id"] == rep_id:
                return analysis, rep
    return None, None


def _ask_panel(profile, analysis: dict | None, rep: dict | None) -> None:
    st.markdown("### Ask Primed")
    st.caption("Questions are answered from the saved measurements. Video stays on this machine.")
    examples = [
        "Which rep changed the most?",
        "Show me the biggest change in this set",
        "Does this repeat on other days at the same load?",
    ]
    cols = st.columns(3)
    pending = st.session_state.get("ask_question") or examples[0]
    for index, example in enumerate(examples):
        if cols[index].button(example, key=f"ask_ex_{index}_{profile.id}"):
            pending = example
            st.session_state["ask_question"] = example
    question = st.text_input("Question", value=pending, key=f"ask_text_{profile.id}")
    if st.button("Ask", type="primary", key=f"ask_go_{profile.id}"):
        reference = None
        if analysis:
            reference = active_reference(_workspace(), analysis["profile_id"], analysis["variant"], analysis["view"])
        bundle = build_bundle(
            _workspace(),
            profile_id=profile.id,
            variant=analysis["variant"] if analysis else profile.default_variant,
            view=analysis["view"] if analysis else profile.default_view,
            selected_set=rep.get("set_label") if rep else None,
            selected_load=(analysis.get("context") or {}).get("load_kg") if analysis else None,
            question=question,
        )
        if reference is None:
            bundle["reference"] = None
        st.session_state.ask = ask_gemini(question, bundle)
    result = st.session_state.get("ask")
    if not result:
        return
    st.write(result["answer"])
    for note in result.get("notes") or []:
        st.caption(note)
    st.caption(f"Source: {'measured records' if result.get('source') != 'gemini' else 'Gemini, checked against the measured records'}.")
    for rep_id in result.get("evidence_rep_ids") or []:
        if st.button(f"Open {rep_id}", key=f"open_evidence_{rep_id}"):
            found, found_rep = _find_rep(rep_id)
            if found and found_rep:
                st.session_state.profile_id = found["profile_id"]
                st.session_state.analysis_id = found["id"]
                st.session_state.rep_id = found_rep["id"]
                st.session_state.view = "review"
                st.rerun()
