import io
import sys
import time
import wave

from fastapi.testclient import TestClient

from stt_lab.api import create_app


class FakeRuntime:
    model_id = None

    async def load(self, spec):
        self.model_id = spec["id"]
        return {"backend": "fake"}

    async def start(self, language, settings):
        pass

    async def push(self, pcm):
        return {"text": ""}

    async def finish(self):
        return {"text": "hej"}

    async def close(self):
        pass


class FakeSpeakerRuntime:
    def __init__(self, data_dir):
        pass

    async def diarize(self, spec, audio_path, profiles, speaker_count, threshold):
        return {
            "model": spec["id"],
            "runtime": {"backend": "fake"},
            "calibration": {"available": False, "reason": "not enough profiles"},
            "detected_speakers": 1,
            "segments": [{
                "start_ms": 0, "end_ms": 3000, "cluster_id": 0,
                "speaker": "Okänd 1", "profile_id": None, "text": "",
            }],
            "timing": {"embedding_and_clustering_seconds": 0.01},
        }

    async def close(self):
        pass


def wav_bytes(seconds=3):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\0\0" * (16000 * seconds))
    return output.getvalue()


def test_upload_profiles_and_speaker_reference(tmp_path):
    stt_catalog = {
        "fake-stt": {
            "id": "fake-stt", "name": "Fake STT", "family": "vosk",
            "checkpoint": "fake", "languages": ["sv"], "python": sys.executable,
            "defaults": {},
        }
    }
    embedding_catalog = {
        "fake-speaker": {
            "id": "fake-speaker", "name": "Fake Speaker", "toolkit": "test",
            "checkpoint": "fake", "revision": "test", "python": sys.executable,
        }
    }
    app = create_app(tmp_path, runtime=FakeRuntime(), catalog=stt_catalog,
                     embedding_catalog=embedding_catalog,
                     speaker_runtime_factory=FakeSpeakerRuntime)
    with TestClient(app) as client:
        uploaded = client.post(
            "/recordings/upload",
            files={"file": ("meeting.wav", wav_bytes(), "audio/wav")},
        )
        assert uploaded.status_code == 201
        recording_id = uploaded.json()["id"]
        assert uploaded.json()["duration"] == 3

        profile = client.post("/speaker-profiles", json={"name": "Ada"})
        assert profile.status_code == 201
        profile_id = profile.json()["id"]
        sample = client.post(
            f"/speaker-profiles/{profile_id}/samples",
            files={"file": ("ada.wav", wav_bytes(), "audio/wav")},
        )
        assert sample.status_code == 201
        assert client.get("/speaker-profiles").json()[0]["samples"][0]["duration"] == 3

        reference = {
            "speakers": [
                {"id": "ada", "label": "Ada", "profile_id": profile_id},
                {"id": "guest", "label": "Gäst", "profile_id": None},
            ],
            "segments": [
                {"start_ms": 0, "end_ms": 1500, "speaker_id": "ada",
                 "text": "hej där"},
                {"start_ms": 1000, "end_ms": 2500, "speaker_id": "guest",
                 "text": "god dag"},
            ],
        }
        response = client.put(
            f"/recordings/{recording_id}/speaker-reference", json=reference)
        assert response.status_code == 200
        assert client.get("/recordings").json()[0]["speaker_reference"] == reference

        started = client.post("/speaker-benchmarks", json={
            "recording_id": recording_id,
            "embedding_models": ["fake-speaker"],
            "stt_model": "fake-stt",
            "language": "sv",
        })
        assert started.status_code == 202
        job_id = started.json()["id"]
        for _ in range(100):
            job = client.get(f"/speaker-benchmarks/{job_id}").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "complete"
        assert job["results"][0]["segments"][0]["text"] == "hej"
        assert job["results"][0]["metrics"]["available"] is True


def test_rejects_bad_audio_and_same_speaker_overlap(tmp_path):
    app = create_app(tmp_path, runtime=FakeRuntime(), catalog={},
                     embedding_catalog={})
    with TestClient(app) as client:
        bad = client.post(
            "/recordings/upload",
            files={"file": ("meeting.mp3", b"not audio", "audio/mpeg")},
        )
        assert bad.status_code == 422

        uploaded = client.post(
            "/recordings/upload",
            files={"file": ("meeting.wav", wav_bytes(), "audio/wav")},
        )
        recording_id = uploaded.json()["id"]
        reference = {
            "speakers": [{"id": "one", "label": "En", "profile_id": None}],
            "segments": [
                {"start_ms": 0, "end_ms": 1500, "speaker_id": "one", "text": "a"},
                {"start_ms": 1000, "end_ms": 2000, "speaker_id": "one", "text": "b"},
            ],
        }
        response = client.put(
            f"/recordings/{recording_id}/speaker-reference", json=reference)
        assert response.status_code == 422
