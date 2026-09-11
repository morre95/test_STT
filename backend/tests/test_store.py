from stt_lab.metrics import accuracy, normalize
from stt_lab.store import Store


def test_accuracy_normalizes_case_and_punctuation():
    assert normalize("Hej, VÄRLDEN!") == "hej världen"
    assert accuracy("Hej världen", "hej världen") == {"wer": 0.0, "cer": 0.0}
    assert accuracy("", "något") == {"wer": None, "cer": None}


def test_recording_and_run_history(tmp_path):
    store = Store(tmp_path)
    recording_id = store.create_recording()
    store.finish_recording(recording_id, 1.25, True)
    store.set_reference(recording_id, "ett test")
    store.save_run({
        "id": "run-1", "recording_id": recording_id,
        "status": "complete", "text": "ett test",
    })
    history = store.history()
    assert history[0]["duration"] == 1.25
    assert history[0]["runs"][0]["wer"] == 0.0


def test_speaker_profiles_references_and_jobs(tmp_path):
    store = Store(tmp_path)
    recording_id = store.create_recording(source="upload", original_name="meeting.wav")
    store.finish_recording(recording_id, 8.0, True)
    profile = store.create_profile("Ada")
    sample = store.add_profile_sample(profile["id"], 3.0, "ada.wav")
    reference = {
        "speakers": [{"id": "speaker-a", "label": "Ada", "profile_id": profile["id"]}],
        "segments": [{"start_ms": 0, "end_ms": 2000, "speaker_id": "speaker-a",
                      "text": "hej"}],
    }
    store.set_speaker_reference(recording_id, reference)
    store.save_speaker_job({
        "id": "job-1", "recording_id": recording_id, "created_at": "2026-01-01",
        "status": "complete", "results": [],
    })

    assert store.profile(profile["id"])["samples"][0]["id"] == sample["id"]
    assert store.history()[0]["speaker_reference"] == reference
    assert store.speaker_jobs()[0]["id"] == "job-1"
