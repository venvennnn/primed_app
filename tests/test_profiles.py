"""The movement library is data. A new profile shows up without a new screen."""

from primed.profiles import (
    EventSpec,
    MetricSpec,
    MovementProfile,
    SegmenterSpec,
    get_profile,
    list_profiles,
    profiles_by_section,
    register_profile,
    unregister_profile,
)


def test_library_has_the_twelve_movements():
    profiles = list_profiles()
    assert len(profiles) == 12
    names = [profile.name for profile in profiles]
    assert names[:7] == [
        "Squat",
        "Deadlift",
        "Bench press",
        "Overhead press",
        "Reverse lunge",
        "Pull-up",
        "Kettlebell swing",
    ]
    assert "Cricket bowling" in names
    sections = dict(profiles_by_section())
    assert len(sections["Strength"]) == 7
    assert len(sections["Sport"]) == 5
    cricket = get_profile("cricket_bowling")
    assert any(event.kind == "candidate" and event.id == "release" for event in cricket.events)
    assert any("Ball release" in item for item in cricket.unavailable)


def test_a_new_profile_registers_without_a_ui_branch():
    profile = MovementProfile(
        id="test_motion",
        name="Test motion",
        section="Skill",
        order=50,
        summary="Registered from a test.",
        variants=(("standard", "Standard"),),
        views=(("side", "Side view"),),
        handedness_required=False,
        context_fields=(),
        events=(EventSpec("start", "Start", "manual"),),
        metrics=(MetricSpec("elbow_min", "Elbow", "deg", "min", "elbow"),),
        frame_readouts=(("elbow", "Elbow", "deg"),),
        trace_series=(("elbow", "Elbow", "deg"),),
        segmenter=SegmenterSpec(mode="manual"),
        phases=None,
        available=("Manual marking",),
        unavailable=("Automatic detection",),
        segmentation_note="Mark the rep manually.",
    )
    register_profile(profile)
    try:
        assert get_profile("test_motion").name == "Test motion"
        assert any(name == "Skill" for name, _items in profiles_by_section())
    finally:
        unregister_profile("test_motion")
