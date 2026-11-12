"""Movement profiles.

Add a movement by registering another ``MovementProfile``. The playground,
review and progress pages iterate the registry, so the screen layout stays
shared.

Segmentation thresholds are part of the profile and of ``SEG_VERSION``. They
describe the 2D signal used to propose reps. A coach can replace those
boundaries. Events that need a ball or a bar are candidates until someone
confirms them.
"""

from __future__ import annotations

from dataclasses import dataclass

from primed.config import PROFILE_VERSION


@dataclass(frozen=True)
class MetricSpec:
    id: str
    label: str
    unit: str
    reducer: str
    series: str | None = None
    event_id: str | None = None
    phase_id: str | None = None
    views: tuple[str, ...] = ("side",)
    note: str = "2D projected measurement."


@dataclass(frozen=True)
class EventSpec:
    id: str
    label: str
    kind: str  # measured, candidate, manual
    note: str = ""


@dataclass(frozen=True)
class PhaseMap:
    toward: str
    toward_label: str
    extreme: str
    extreme_label: str
    returning: str
    returning_label: str
    start_event: str
    extreme_event: str
    end_event: str


@dataclass(frozen=True)
class SegmenterSpec:
    mode: str  # dip, rise, peak, manual
    signal: str | None = None
    arm: float = 0.0
    enter: float = 0.0
    gate: float = 0.0
    recover: float = 0.0
    min_prominence: float = 25.0
    min_duration_s: float = 0.40
    max_duration_s: float = 8.0
    band: float = 8.0
    pre_s: float = 0.70
    post_s: float = 0.50
    peak_event: str | None = None


@dataclass(frozen=True)
class MovementProfile:
    id: str
    name: str
    section: str
    order: int
    summary: str
    variants: tuple[tuple[str, str], ...]
    views: tuple[tuple[str, str], ...]
    handedness_required: bool
    context_fields: tuple[tuple[str, str, str], ...]
    events: tuple[EventSpec, ...]
    metrics: tuple[MetricSpec, ...]
    frame_readouts: tuple[tuple[str, str, str], ...]
    trace_series: tuple[tuple[str, str, str], ...]
    segmenter: SegmenterSpec
    phases: PhaseMap | None
    available: tuple[str, ...]
    unavailable: tuple[str, ...]
    segmentation_note: str
    version: str = PROFILE_VERSION

    @property
    def default_variant(self) -> str:
        return self.variants[0][0]

    @property
    def default_view(self) -> str:
        return self.views[0][0]

    def variant_label(self, variant_id: str) -> str:
        for key, label in self.variants:
            if key == variant_id:
                return label
        return variant_id

    def view_label(self, view_id: str) -> str:
        for key, label in self.views:
            if key == view_id:
                return label
        return view_id

    def metric(self, metric_id: str) -> MetricSpec | None:
        for item in self.metrics:
            if item.id == metric_id:
                return item
        return None

    def event(self, event_id: str) -> EventSpec | None:
        for item in self.events:
            if item.id == event_id:
                return item
        return None


def _dip(signal: str, arm: float, enter: float, gate: float, recover: float, prominence: float = 25.0) -> SegmenterSpec:
    return SegmenterSpec(
        mode="dip",
        signal=signal,
        arm=arm,
        enter=enter,
        gate=gate,
        recover=recover,
        min_prominence=prominence,
    )


def _rise(signal: str, arm: float, enter: float, gate: float, recover: float, prominence: float = 25.0) -> SegmenterSpec:
    return SegmenterSpec(
        mode="rise",
        signal=signal,
        arm=arm,
        enter=enter,
        gate=gate,
        recover=recover,
        min_prominence=prominence,
    )


def _phases(
    toward: str,
    toward_label: str,
    extreme: str,
    extreme_label: str,
    returning: str,
    returning_label: str,
    start_event: str,
    extreme_event: str,
    end_event: str,
) -> PhaseMap:
    return PhaseMap(
        toward=toward,
        toward_label=toward_label,
        extreme=extreme,
        extreme_label=extreme_label,
        returning=returning,
        returning_label=returning_label,
        start_event=start_event,
        extreme_event=extreme_event,
        end_event=end_event,
    )


_SIDE = (("side", "Side view"),)
_SIDE_FRONT = (("side", "Side view"), ("front", "Front view"))
_LOAD = (("load_kg", "Load (kg)", "number"),)
_DRILL = (("drill", "Drill", "text"),)
_ANGLE = "2D projected angle in degrees."
_TIME = "Raw timing in seconds from the clip timestamps. Phase alignment is not used."


def _builtin_profiles() -> list[MovementProfile]:
    squat = MovementProfile(
        id="squat",
        name="Squat",
        section="Strength",
        order=1,
        summary="Side-view knee and hip angles, trunk tilt, and phase timing.",
        variants=(("air", "Air squat"), ("back", "Back squat"), ("front", "Front squat"), ("goblet", "Goblet squat")),
        views=_SIDE + (("front", "Front view"),),
        handedness_required=False,
        context_fields=_LOAD,
        events=(
            EventSpec("standing_start", "Standing", "measured", "Knee angle is extended before the descent."),
            EventSpec("bottom", "Bottom", "measured", "Lowest knee angle in the rep."),
            EventSpec("standing_end", "Standing", "measured", "Knee angle has returned to extension."),
        ),
        metrics=(
            MetricSpec("bottom_knee_angle", "Bottom knee angle", "deg", "min", "knee", note=_ANGLE),
            MetricSpec("bottom_hip_angle", "Bottom hip angle", "deg", "at_event", "hip", event_id="bottom", note=_ANGLE),
            MetricSpec(
                "hip_drop",
                "Bottom hip drop",
                "torso",
                "hip_drop",
                note="2D hip drop divided by standing torso length. Not a metric distance.",
            ),
            MetricSpec("trunk_tilt", "Trunk tilt", "deg", "at_event", "trunk_tilt", event_id="bottom", note="Positive is forward lean."),
            MetricSpec("descent_s", "Descent", "s", "duration", phase_id="descent", note=_TIME),
            MetricSpec("ascent_s", "Ascent", "s", "duration", phase_id="ascent", note=_TIME),
        ),
        frame_readouts=(("knee", "Knee angle", "deg"), ("hip", "Hip angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("knee", "Knee angle", "deg"), ("hip", "Hip angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=_dip("knee", arm=160, enter=150, gate=125, recover=158),
        phases=_phases("descent", "Descent", "bottom", "Bottom", "ascent", "Ascent", "standing_start", "bottom", "standing_end"),
        available=(
            "Automatic rep detection from the side-view knee angle",
            "Manual rep and phase edits",
            "Knee angle, hip angle, trunk tilt, hip-drop proxy, descent and ascent time",
        ),
        unavailable=("Bar speed and bar path (no object tracker or scale)", "3D joint angles"),
        segmentation_note=(
            "A rep is a dip in the more visible knee angle. Detection starts from extension "
            "at or above 160°, requires a bottom at or below 125° and a return to 158°. "
            "The dip must be at least 25° and between 0.4 s and 8 s. The signal is median-smoothed "
            "only for detection. Angles and times are read from the raw samples. "
            "Positive trunk tilt is forward lean toward the selected facing direction."
        ),
    )
    deadlift = MovementProfile(
        id="deadlift",
        name="Deadlift",
        section="Strength",
        order=2,
        summary="Side-view hip and knee angles, trunk tilt, and phase timing.",
        variants=(("conventional", "Conventional"), ("sumo", "Sumo"), ("romanian", "Romanian")),
        views=_SIDE,
        handedness_required=False,
        context_fields=_LOAD,
        events=(
            EventSpec("setup", "Setup", "measured", "Hip is flexed before the pull."),
            EventSpec("lift_off", "Lift-off", "candidate", "Hip angle starts to extend. The bar is not tracked."),
            EventSpec("knee_pass", "Knee pass", "candidate", "Knee angle crosses 150° during the pull. The bar is not tracked."),
            EventSpec("lockout", "Lockout", "measured", "Most extended hip angle in the rep."),
            EventSpec("lowering", "Lowering", "measured", "Hip angle has returned toward the setup."),
        ),
        metrics=(
            MetricSpec("setup_hip_angle", "Setup hip angle", "deg", "at_event", "hip", event_id="setup", note=_ANGLE),
            MetricSpec("lockout_hip_angle", "Lockout hip angle", "deg", "at_event", "hip", event_id="lockout", note=_ANGLE),
            MetricSpec("lockout_knee_angle", "Lockout knee angle", "deg", "at_event", "knee", event_id="lockout", note=_ANGLE),
            MetricSpec("setup_trunk_tilt", "Setup trunk tilt", "deg", "at_event", "trunk_tilt", event_id="setup", note="Positive is forward lean."),
            MetricSpec("ascent_s", "Ascent", "s", "duration", phase_id="ascent", note=_TIME),
            MetricSpec("lowering_s", "Lowering", "s", "duration", phase_id="lowering", note=_TIME),
        ),
        frame_readouts=(("hip", "Hip angle", "deg"), ("knee", "Knee angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("hip", "Hip angle", "deg"), ("knee", "Knee angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=_rise("hip", arm=120, enter=135, gate=155, recover=125),
        phases=_phases("ascent", "Ascent", "lockout", "Lockout", "lowering", "Lowering", "setup", "lockout", "lowering"),
        available=(
            "Automatic rep detection from the side-view hip angle",
            "Manual edits",
            "Hip angle, knee angle, trunk tilt, ascent and lowering time",
        ),
        unavailable=("Bar path and bar speed (no object tracker or scale)", "Confirmed knee-pass without a coach marker"),
        segmentation_note=(
            "A rep is a rise in the hip angle from a flexed setup at or below 120°, through "
            "155° at lockout, and back to 125°. Lift-off and knee pass stay candidates because "
            "the bar is not tracked. Ascent time uses the hip-angle phases and the clip timestamps."
        ),
    )
    bench = MovementProfile(
        id="bench_press",
        name="Bench press",
        section="Strength",
        order=3,
        summary="Elbow angle and phase timing where the arm is visible.",
        variants=(("flat", "Flat"), ("incline", "Incline"), ("dumbbell", "Dumbbell")),
        views=_SIDE,
        handedness_required=False,
        context_fields=_LOAD,
        events=(
            EventSpec("lockout_start", "Lockout", "measured"),
            EventSpec("bottom", "Bottom", "measured", "Most flexed elbow in the rep."),
            EventSpec("lockout_end", "Lockout", "measured"),
        ),
        metrics=(
            MetricSpec("bottom_elbow_angle", "Bottom elbow angle", "deg", "min", "elbow", note=_ANGLE),
            MetricSpec("descent_s", "Descent", "s", "duration", phase_id="descent", note=_TIME),
            MetricSpec("press_s", "Press", "s", "duration", phase_id="press", note=_TIME),
        ),
        frame_readouts=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("elbow", "Elbow angle", "deg"),),
        segmenter=_dip("elbow", arm=155, enter=145, gate=100, recover=150, prominence=30),
        phases=_phases("descent", "Descent", "bottom", "Bottom", "press", "Press", "lockout_start", "bottom", "lockout_end"),
        available=("Elbow angle and descent/press time when the arm is visible", "Manual rep marking"),
        unavailable=("Bar path", "Grip width"),
        segmentation_note=(
            "A rep is a dip in elbow angle from lockout at or above 155° to a bottom at or below 100°, "
            "then back to 150°. If the elbows are hidden, mark the rep manually."
        ),
    )
    overhead = MovementProfile(
        id="overhead_press",
        name="Overhead press",
        section="Strength",
        order=4,
        summary="Elbow angle, trunk tilt, and phase timing.",
        variants=(("standing", "Standing"), ("seated", "Seated"), ("dumbbell", "Dumbbell")),
        views=_SIDE,
        handedness_required=False,
        context_fields=_LOAD,
        events=(
            EventSpec("start", "Start", "measured"),
            EventSpec("lockout", "Lockout", "measured"),
            EventSpec("lowering_end", "Lowering", "measured"),
        ),
        metrics=(
            MetricSpec("lockout_elbow_angle", "Lockout elbow angle", "deg", "at_event", "elbow", event_id="lockout", note=_ANGLE),
            MetricSpec("lockout_trunk_tilt", "Trunk tilt", "deg", "at_event", "trunk_tilt", event_id="lockout", note="Positive is forward lean."),
            MetricSpec("press_s", "Press", "s", "duration", phase_id="press", note=_TIME),
            MetricSpec("lowering_s", "Lowering", "s", "duration", phase_id="lowering", note=_TIME),
        ),
        frame_readouts=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=_rise("elbow", arm=120, enter=140, gate=160, recover=125),
        phases=_phases("press", "Press", "lockout", "Lockout", "lowering", "Lowering", "start", "lockout", "lowering_end"),
        available=("Elbow angle, trunk tilt, press and lowering time", "Manual rep marking"),
        unavailable=("Bar path", "3D shoulder angle"),
        segmentation_note=(
            "A rep is a rise in elbow angle from a rack position at or below 120° to lockout at or above 160°, "
            "then a return to 125°."
        ),
    )
    lunge = MovementProfile(
        id="reverse_lunge",
        name="Reverse lunge",
        section="Strength",
        order=5,
        summary="Front-leg knee angle, trunk tilt, and phase timing.",
        variants=(("reverse", "Reverse"), ("walking", "Walking")),
        views=_SIDE,
        handedness_required=False,
        context_fields=_LOAD + (("front_leg", "Front leg", "side"),),
        events=(
            EventSpec("standing_start", "Standing", "measured"),
            EventSpec("bottom", "Bottom", "measured"),
            EventSpec("standing_end", "Standing", "measured"),
        ),
        metrics=(
            MetricSpec("bottom_knee_angle", "Front-leg knee angle", "deg", "min", "knee", note=_ANGLE + " Uses the selected front leg when set."),
            MetricSpec("trunk_tilt", "Trunk tilt", "deg", "at_event", "trunk_tilt", event_id="bottom", note="Positive is forward lean."),
            MetricSpec("descent_s", "Step down", "s", "duration", phase_id="descent", note=_TIME),
            MetricSpec("return_s", "Return", "s", "duration", phase_id="return", note=_TIME),
        ),
        frame_readouts=(("knee", "Front-leg knee", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("knee", "Front-leg knee", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=_dip("knee", arm=160, enter=145, gate=110, recover=155),
        phases=_phases("descent", "Step down", "bottom", "Bottom", "return", "Return", "standing_start", "bottom", "standing_end"),
        available=("Front-leg knee angle, trunk tilt, and phase time", "Manual rep marking"),
        unavailable=("Step length (no scale)", "Which foot is the front foot, unless you set it"),
        segmentation_note=(
            "A rep is a dip in the front-leg knee angle. Set the front leg when you know it. "
            "Otherwise the leg with the larger range is used and the rep is marked uncertain."
        ),
    )
    pullup = MovementProfile(
        id="pull_up",
        name="Pull-up",
        section="Strength",
        order=6,
        summary="Elbow angle and phase timing.",
        variants=(("pronated", "Pronated"), ("supinated", "Supinated"), ("neutral", "Neutral")),
        views=_SIDE_FRONT,
        handedness_required=False,
        context_fields=(),
        events=(
            EventSpec("bottom_start", "Bottom", "measured", "Elbow is extended."),
            EventSpec("top", "Top", "measured", "Most flexed elbow. Chin clearance is not detected."),
            EventSpec("bottom_end", "Bottom", "measured"),
        ),
        metrics=(
            MetricSpec("top_elbow_angle", "Top elbow angle", "deg", "min", "elbow", views=("side", "front"), note=_ANGLE),
            MetricSpec("ascent_s", "Ascent", "s", "duration", phase_id="ascent", views=("side", "front"), note=_TIME),
            MetricSpec("descent_s", "Descent", "s", "duration", phase_id="descent", views=("side", "front"), note=_TIME),
        ),
        frame_readouts=(("elbow", "Elbow angle", "deg"),),
        trace_series=(("elbow", "Elbow angle", "deg"),),
        segmenter=_dip("elbow", arm=155, enter=140, gate=80, recover=150, prominence=30),
        phases=_phases("ascent", "Ascent", "top", "Top", "descent", "Descent", "bottom_start", "top", "bottom_end"),
        available=("Elbow angle and ascent/descent time from the side or front", "Manual rep marking"),
        unavailable=("Chin over the bar", "Added weight, unless you note it outside the measurement"),
        segmentation_note=(
            "A rep is a dip in elbow angle from a hang at or above 155° to a flexed top at or below 80°, "
            "then back to 150°. Clearing the bar is not detected."
        ),
    )
    swing = MovementProfile(
        id="kettlebell_swing",
        name="Kettlebell swing",
        section="Strength",
        order=7,
        summary="Hip-angle proxy and cadence from the side.",
        variants=(("russian", "Russian"), ("american", "American"), ("one_arm", "One-arm")),
        views=_SIDE,
        handedness_required=False,
        context_fields=_LOAD,
        events=(
            EventSpec("hinge", "Hinge", "measured"),
            EventSpec("peak", "Peak", "measured", "Most extended hip angle. The bell is not tracked."),
            EventSpec("return", "Return", "measured"),
        ),
        metrics=(
            MetricSpec("hinge_hip_angle", "Hinge hip angle", "deg", "min", "hip", note=_ANGLE + " Proxy for the hinge, not bell height."),
            MetricSpec("peak_hip_angle", "Peak hip angle", "deg", "max", "hip", note=_ANGLE),
            MetricSpec("cadence_rpm", "Cadence", "rpm", "cadence", note="Cycles per minute from this rep's own duration."),
        ),
        frame_readouts=(("hip", "Hip angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("hip", "Hip angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=_rise("hip", arm=130, enter=145, gate=160, recover=135),
        phases=_phases("extension", "Extension", "peak", "Peak", "return", "Return", "hinge", "peak", "return"),
        available=("Hip-angle proxy, trunk tilt, and cadence", "Manual rep marking"),
        unavailable=("Bell height and bell path", "A confirmed peak of the bell"),
        segmentation_note=(
            "A cycle is a rise in hip angle from a hinge at or below 130° to extension at or above 160°, "
            "then a return to 135°. Cadence is 60 divided by that cycle's duration."
        ),
    )
    golf = MovementProfile(
        id="golf_swing",
        name="Golf swing",
        section="Sport",
        order=8,
        summary="Phase timing and observable 2D pose once you mark the swing.",
        variants=(("full", "Full swing"), ("pitch", "Pitch"), ("chip", "Chip")),
        views=(("down_the_line", "Down the line"), ("face_on", "Face on")),
        handedness_required=True,
        context_fields=(("club", "Club", "text"),),
        events=(
            EventSpec("address", "Address", "manual"),
            EventSpec("top", "Top", "manual"),
            EventSpec("impact", "Impact candidate", "candidate", "Confirm impact. The club and ball are not tracked."),
            EventSpec("follow_through", "Follow-through", "manual"),
        ),
        metrics=(
            MetricSpec("elbow_min", "Minimum elbow angle", "deg", "min", "elbow", views=("down_the_line", "face_on"), note=_ANGLE),
            MetricSpec("backswing_s", "Backswing", "s", "duration", phase_id="backswing", views=("down_the_line", "face_on"), note=_TIME),
            MetricSpec("downswing_s", "Downswing", "s", "duration", phase_id="downswing", views=("down_the_line", "face_on"), note=_TIME),
        ),
        frame_readouts=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=SegmenterSpec(mode="manual"),
        phases=_phases("backswing", "Backswing", "top", "Top", "downswing", "Downswing", "address", "top", "impact"),
        available=("Manual rep and event marking", "Elbow angle on the marked interval", "Backswing and downswing time after the events are confirmed"),
        unavailable=("Club path", "Impact, until you confirm it", "Ball speed"),
        segmentation_note=(
            "Automatic swing detection is not available. Mark one rep, then place address, top, "
            "impact and follow-through. Impact stays a candidate until you confirm it."
        ),
    )
    cricket = MovementProfile(
        id="cricket_bowling",
        name="Cricket bowling",
        section="Sport",
        order=9,
        summary="Elbow, knee and trunk traces, with a candidate delivery window.",
        variants=(("pace", "Pace"), ("spin", "Spin")),
        views=_SIDE,
        handedness_required=True,
        context_fields=_DRILL,
        events=(
            EventSpec("action_start", "Action start", "candidate", "Start of the detected window, not a measured run-up."),
            EventSpec("delivery_stride", "Delivery stride", "candidate", "Lowest front-knee angle in the window. Confirm the stride."),
            EventSpec("release", "Release candidate", "candidate", "Fastest 2D wrist movement of the bowling arm. The ball is not tracked."),
            EventSpec("follow_through", "Follow-through", "candidate"),
        ),
        metrics=(
            MetricSpec("bowling_elbow", "Bowling elbow", "deg", "min", "elbow", note=_ANGLE + " Minimum on the marked delivery."),
            MetricSpec("front_knee", "Front knee", "deg", "min", "knee", note=_ANGLE + " Front leg is opposite the bowling arm unless you set it."),
            MetricSpec("trunk_tilt", "Trunk tilt", "deg", "median", "trunk_tilt", note="Positive is forward lean."),
            MetricSpec("delivery_s", "Delivery", "s", "duration", phase_id="delivery", note=_TIME),
            MetricSpec("follow_s", "Follow-through", "s", "duration", phase_id="follow_through", note=_TIME),
        ),
        frame_readouts=(
            ("elbow_left", "Left elbow", "deg"),
            ("knee_right", "Right knee", "deg"),
            ("trunk_tilt", "Trunk tilt", "deg"),
        ),
        trace_series=(
            ("elbow_left", "Left elbow", "deg"),
            ("elbow", "Bowling elbow", "deg"),
            ("knee_right", "Right knee", "deg"),
            ("knee", "Front knee", "deg"),
            ("trunk_tilt", "Trunk tilt", "deg"),
        ),
        segmenter=SegmenterSpec(mode="peak", signal="wrist_speed", pre_s=0.80, post_s=0.50, peak_event="release"),
        phases=_phases(
            "delivery",
            "Delivery",
            "release",
            "Release candidate",
            "follow_through",
            "Follow-through",
            "delivery_stride",
            "release",
            "follow_through",
        ),
        available=(
            "Left and right elbow, knee and trunk tilt on visible frames",
            "A candidate delivery window from bowling-arm wrist speed",
            "Manual rep and event correction",
        ),
        unavailable=(
            "Ball release, until you confirm the candidate",
            "Ball speed and line (no object tracker or scale)",
            "Run-up speed",
        ),
        segmentation_note=(
            "The bowling arm follows the selected handedness and the front leg is the other side. "
            "A candidate window is placed around the fastest 2D wrist movement. That peak is a release "
            "candidate, not a detected ball. Elbow, knee and trunk values on the window are measured. "
            "Delivery time stays unavailable until you confirm the stride and the release. "
            "The frame strip shows left elbow, right knee and trunk tilt."
        ),
    )
    tennis = MovementProfile(
        id="tennis_serve",
        name="Tennis serve",
        section="Sport",
        order=10,
        summary="Elbow angle and timing. Contact is a candidate until you confirm it.",
        variants=(("flat", "Flat"), ("slice", "Slice"), ("kick", "Kick")),
        views=(("side", "Side view"), ("rear", "Rear view")),
        handedness_required=True,
        context_fields=_DRILL,
        events=(
            EventSpec("setup", "Setup", "manual"),
            EventSpec("trophy", "Trophy position", "manual"),
            EventSpec("contact", "Contact candidate", "candidate", "The ball is not tracked."),
            EventSpec("follow_through", "Follow-through", "manual"),
        ),
        metrics=(
            MetricSpec("elbow_max", "Peak elbow angle", "deg", "max", "elbow", views=("side", "rear"), note=_ANGLE),
            MetricSpec("trophy_s", "To trophy", "s", "duration", phase_id="to_trophy", views=("side", "rear"), note=_TIME),
            MetricSpec("contact_s", "To contact", "s", "duration", phase_id="to_contact", views=("side", "rear"), note=_TIME),
        ),
        frame_readouts=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=SegmenterSpec(mode="manual"),
        phases=_phases("to_trophy", "To trophy", "trophy", "Trophy", "to_contact", "To contact", "setup", "trophy", "contact"),
        available=("Manual marking", "Elbow angle on the marked rep", "Phase time after events are confirmed"),
        unavailable=("Ball contact, until you confirm it", "Racket speed"),
        segmentation_note="Automatic serve detection is not available. Mark the rep and confirm contact before using its timing.",
    )
    free_throw = MovementProfile(
        id="free_throw",
        name="Free throw",
        section="Sport",
        order=11,
        summary="Knee and elbow angles, plus phase timing you confirm.",
        variants=(("standard", "Standard"),),
        views=_SIDE,
        handedness_required=True,
        context_fields=_DRILL,
        events=(
            EventSpec("setup", "Setup", "manual"),
            EventSpec("extension", "Extension", "manual"),
            EventSpec("release", "Release candidate", "candidate", "The ball is not tracked."),
            EventSpec("follow_through", "Follow-through", "manual"),
        ),
        metrics=(
            MetricSpec("knee_angle", "Knee angle", "deg", "min", "knee", note=_ANGLE),
            MetricSpec("elbow_angle", "Elbow angle", "deg", "max", "elbow", note=_ANGLE),
            MetricSpec("extension_s", "Extension", "s", "duration", phase_id="extension", note=_TIME),
        ),
        frame_readouts=(("knee", "Knee angle", "deg"), ("elbow", "Elbow angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("knee", "Knee angle", "deg"), ("elbow", "Elbow angle", "deg")),
        segmenter=SegmenterSpec(mode="manual"),
        phases=_phases("extension", "Extension", "release", "Release candidate", "follow_through", "Follow-through", "setup", "release", "follow_through"),
        available=("Manual marking", "Knee and elbow angles on the marked rep"),
        unavailable=("Ball release, until you confirm it", "Release height"),
        segmentation_note="Automatic free-throw detection is not available. Mark the shot, then confirm the release candidate.",
    )
    pitch = MovementProfile(
        id="baseball_pitch",
        name="Baseball pitch",
        section="Sport",
        order=12,
        summary="Observable pose angles and timing. Release is a candidate.",
        variants=(("fastball", "Fastball"), ("breaking", "Breaking ball"), ("changeup", "Changeup")),
        views=_SIDE,
        handedness_required=True,
        context_fields=_DRILL,
        events=(
            EventSpec("setup", "Setup", "manual"),
            EventSpec("stride", "Stride", "manual"),
            EventSpec("release", "Release candidate", "candidate", "The ball is not tracked."),
            EventSpec("follow_through", "Follow-through", "manual"),
        ),
        metrics=(
            MetricSpec("elbow_angle", "Elbow angle", "deg", "max", "elbow", note=_ANGLE),
            MetricSpec("knee_angle", "Knee angle", "deg", "min", "knee", note=_ANGLE),
            MetricSpec("trunk_tilt", "Trunk tilt", "deg", "median", "trunk_tilt", note="Positive is forward lean."),
            MetricSpec("stride_s", "To release", "s", "duration", phase_id="to_release", note=_TIME),
        ),
        frame_readouts=(("elbow", "Elbow angle", "deg"), ("knee", "Knee angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        trace_series=(("elbow", "Elbow angle", "deg"), ("knee", "Knee angle", "deg"), ("trunk_tilt", "Trunk tilt", "deg")),
        segmenter=SegmenterSpec(mode="manual"),
        phases=_phases("to_stride", "To stride", "stride", "Stride", "to_release", "To release", "setup", "stride", "release"),
        available=("Manual marking", "Elbow, knee and trunk tilt on the marked rep"),
        unavailable=("Ball release, until you confirm it", "Pitch speed"),
        segmentation_note="Automatic pitch detection is not available. Mark the pitch and confirm release before using its timing.",
    )
    return [squat, deadlift, bench, overhead, lunge, pullup, swing, golf, cricket, tennis, free_throw, pitch]


_REGISTRY: dict[str, MovementProfile] = {profile.id: profile for profile in _builtin_profiles()}


def register_profile(profile: MovementProfile) -> None:
    _REGISTRY[profile.id] = profile


def unregister_profile(profile_id: str) -> None:
    _REGISTRY.pop(profile_id, None)


def get_profile(profile_id: str) -> MovementProfile:
    try:
        return _REGISTRY[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown movement profile '{profile_id}'.") from exc


def list_profiles() -> list[MovementProfile]:
    return sorted(_REGISTRY.values(), key=lambda profile: profile.order)


def profiles_by_section() -> list[tuple[str, list[MovementProfile]]]:
    grouped: dict[str, list[MovementProfile]] = {}
    for profile in list_profiles():
        grouped.setdefault(profile.section, []).append(profile)
    order = ["Strength", "Sport"]
    sections = [(name, grouped[name]) for name in order if name in grouped]
    for name, items in grouped.items():
        if name not in order:
            sections.append((name, items))
    return sections
