import numpy as np
import pytest

from stt_lab.speaker_metrics import (
    concatenated_permutation_wer,
    diarization,
    evaluate,
    named_speaker_wer,
)
from stt_lab.speaker_pipeline import _calibrate, _turns, _windows

REFERENCE = {
    "speakers": [
        {"id": "alice", "label": "Alice", "profile_id": "profile-a"},
        {"id": "visitor", "label": "Besökare", "profile_id": None},
    ],
    "segments": [
        {"start_ms": 0, "end_ms": 1000, "speaker_id": "alice", "text": "hej där"},
        {"start_ms": 1000, "end_ms": 2000, "speaker_id": "visitor", "text": "god dag"},
    ],
}


def test_perfect_diarization_is_permutation_invariant():
    hypothesis = [
        {"start_ms": 0, "end_ms": 1000, "cluster_id": 9, "text": "hej där",
         "profile_id": "profile-a"},
        {"start_ms": 1000, "end_ms": 2000, "cluster_id": 4, "text": "god dag",
         "profile_id": None},
    ]
    metrics = evaluate(REFERENCE, hypothesis, 2.0)
    assert metrics["der"]["der"] == pytest.approx(0)
    assert metrics["cpwer"] == pytest.approx(0)
    assert metrics["speaker_attributed_wer"] == pytest.approx(0)
    assert metrics["identity"]["accuracy"] == pytest.approx(1)


def test_named_wer_penalizes_wrong_known_identity():
    hypothesis = [
        {"start_ms": 0, "end_ms": 1000, "cluster_id": 0, "text": "hej där",
         "profile_id": "wrong-profile"},
        {"start_ms": 1000, "end_ms": 2000, "cluster_id": 1, "text": "god dag",
         "profile_id": None},
    ]
    speakers = {item["id"]: item for item in REFERENCE["speakers"]}
    assert concatenated_permutation_wer(REFERENCE["segments"], hypothesis) == pytest.approx(0)
    assert named_speaker_wer(REFERENCE["segments"], hypothesis, speakers) == pytest.approx(1)


def test_overlap_can_be_included_or_excluded():
    reference = [
        {"start_ms": 0, "end_ms": 2000, "speaker_id": "a", "text": "a"},
        {"start_ms": 1000, "end_ms": 2000, "speaker_id": "b", "text": "b"},
    ]
    hypothesis = [
        {"start_ms": 0, "end_ms": 2000, "cluster_id": 0, "text": "a"},
    ]
    without = diarization(reference, hypothesis, 2000, collar_ms=0, skip_overlap=True)
    with_overlap = diarization(reference, hypothesis, 2000, collar_ms=0, skip_overlap=False)
    assert without["der"] == pytest.approx(0)
    assert with_overlap["der"] > 0


def test_profile_calibration_is_model_specific_and_overridable():
    profiles = [{"id": "a"}, {"id": "b"}]
    vectors = {
        "a": [np.array([1.0, 0.0]), np.array([0.99, 0.01])],
        "b": [np.array([0.0, 1.0]), np.array([0.01, 0.99])],
    }
    automatic, centroids = _calibrate(profiles, vectors, None)
    manual, _ = _calibrate(profiles, vectors, 0.75)
    assert automatic["available"] is True
    assert set(centroids) == {"a", "b"}
    assert manual["threshold"] == 0.75
    assert manual["manual_override"] is True


def test_common_windows_are_merged_into_turns():
    windows = _windows([(0.0, 3.0)], 3.0)
    turns = _turns([(0.0, 3.0)], windows, np.array([0, 0, 1]))
    assert turns[0]["start_ms"] == 0
    assert turns[-1]["end_ms"] == 3000
    assert [item["cluster_id"] for item in turns] == [0, 1]
