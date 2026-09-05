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
