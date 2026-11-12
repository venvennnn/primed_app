"""Ask Primed stays inside the measured records."""

import json

from primed.ask import answer_question, ask_gemini, build_bundle, validate_model_answer
from primed.fixture import build_synthetic_workspace


def _bundle():
    workspace = build_synthetic_workspace()
    return workspace, build_bundle(
        workspace,
        profile_id="squat",
        variant="air",
        view="side",
        selected_set="Set 4",
        selected_load=60,
        question="Which rep changed the most?",
    )


def test_biggest_change_uses_a_real_rep():
    _workspace, bundle = _bundle()
    result = answer_question("Which rep changed the most?", bundle)
    assert result["source"] == "deterministic"
    assert result["findings"]
    finding = result["findings"][0]
    assert finding["rep_id"] in {rep["rep_id"] for rep in bundle["reps"]}
    assert finding["metric_id"] in bundle["reps"][0]["metrics"] or finding["metric_id"] in (
        bundle["reps"][0]["signed"]
    )
    matched = next(rep for rep in bundle["reps"] if rep["rep_id"] == finding["rep_id"])
    assert abs(finding["value"] - matched["signed"][finding["metric_id"]]) < 0.05
    assert finding["rep_id"] in result["evidence_rep_ids"]


def test_this_set_does_not_cite_another_set():
    _workspace, bundle = _bundle()
    result = answer_question("Show me the biggest change in this set", bundle)
    assert result["evidence_rep_ids"]
    for rep_id in result["evidence_rep_ids"]:
        rep = next(item for item in bundle["reps"] if item["rep_id"] == rep_id)
        assert rep["set_label"] == "Set 4"


def test_repeat_question_reports_sample_size_without_a_recommendation():
    _workspace, bundle = _bundle()
    result = answer_question("Does this repeat on other days at the same load?", bundle)
    text = result["answer"].lower()
    assert "sample size" in text or "not enough" in text
    assert "recommend" not in text
    assert "improv" not in text


def test_bad_ids_and_values_are_removed():
    _workspace, bundle = _bundle()
    real = bundle["reps"][0]
    raw = {
        "answer": f"Look at rep_deadbeef0000 and {real['rep_id']}.",
        "findings": [
            {
                "rep_id": "rep_deadbeef0000",
                "metric_id": "bottom_knee_angle",
                "value": 1,
                "unit": "deg",
                "timestamp_s": 0.2,
            },
            {
                "rep_id": real["rep_id"],
                "metric_id": "bottom_knee_angle",
                "value": 999,
                "unit": "deg",
                "timestamp_s": real["timestamp_s"],
            },
            {
                "rep_id": real["rep_id"],
                "metric_id": "bottom_knee_angle",
                "value": real["signed"]["bottom_knee_angle"],
                "unit": "deg",
                "timestamp_s": real["timestamp_s"],
            },
        ],
        "evidence_rep_ids": ["rep_deadbeef0000", real["rep_id"]],
    }
    checked = validate_model_answer(raw, bundle)
    assert checked["source"] == "deterministic" or all(item["rep_id"] == real["rep_id"] for item in checked["findings"])
    blob = json.dumps(checked)
    assert "rep_deadbeef0000" not in blob
    assert "999" not in blob


def test_bundle_has_no_media_or_secrets():
    _workspace, bundle = _bundle()
    blob = json.dumps(bundle)
    assert "GEMINI" not in blob
    assert "path" not in blob
    assert "mp4" not in blob


def test_missing_key_uses_the_measured_answer(monkeypatch):
    _workspace, bundle = _bundle()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = ask_gemini("Which rep changed the most?", bundle)
    assert result["source"] == "deterministic"
    assert result["findings"]
    assert any("no Gemini" in note for note in result["notes"])
