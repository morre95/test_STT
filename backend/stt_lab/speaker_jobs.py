from __future__ import annotations

import asyncio
import contextlib
import time
import wave
from datetime import UTC, datetime
from uuid import uuid4

from .metrics import accuracy
from .speaker_metrics import evaluate
from .speaker_runtime import SpeakerRuntime


class SpeakerJobs:
    def __init__(self, store, embedding_catalog, stt_catalog, stt_runtime, resource_lock,
                 runtime_factory=SpeakerRuntime):
        self.store = store
        self.embedding_catalog = embedding_catalog
        self.stt_catalog = stt_catalog
        self.stt_runtime = stt_runtime
        self.resource_lock = resource_lock
        self.runtime_factory = runtime_factory
        self.tasks = {}

    def submit(self, request):
        ident = str(uuid4())
        job = {"id": ident, "recording_id": request["recording_id"],
               "created_at": datetime.now(UTC).isoformat(), "status": "queued",
               "progress": 0.0, "stage": "I kö", "request": request, "results": []}
        self.store.save_speaker_job(job)
        self.tasks[ident] = asyncio.create_task(self._run(ident))
        return job

    def _update(self, job, **changes):
        job.update(changes)
        self.store.save_speaker_job(job)

    def _profiles(self):
        result = []
        for profile in self.store.profiles():
            samples = [
                {**sample,
                 "path": str(self.store.profile_sample_path(profile["id"], sample["id"]))}
                for sample in profile["samples"]
            ]
            result.append({**profile, "samples": samples})
        return result

    async def _transcribe(self, audio_path, segments, spec, language):
        await self.stt_runtime.load(spec)
        output = []
        with wave.open(str(audio_path), "rb") as source:
            rate = source.getframerate()
            total_frames = source.getnframes()
            for original in segments:
                segment = dict(original)
                start = max(0, round(segment["start_ms"] * rate / 1000))
                end = max(start, round(segment["end_ms"] * rate / 1000))
                start, end = min(start, total_frames), min(end, total_frames)
                source.setpos(start)
                remaining = end - start
                await self.stt_runtime.start(language, spec["defaults"])
                while remaining > 0:
                    count = min(1600, remaining)
                    pcm = source.readframes(count)
                    if not pcm:
                        break
                    await self.stt_runtime.push(pcm)
                    remaining -= count
                final = await self.stt_runtime.finish()
                segment["text"] = final.get("text", "")
                output.append(segment)
        return output

    async def _run(self, ident):
        job = self.store.speaker_job(ident)
        request = job["request"]
        runtime = self.runtime_factory(self.store.root)
        try:
            async with self.resource_lock:
                recording = self.store.recording(job["recording_id"])
                audio_path = self.store.audio_path(job["recording_id"])
                reference = self.store.speaker_reference(job["recording_id"])
                profiles = self._profiles()
                total_models = len(request["embedding_models"])
                self._update(job, status="running", stage="Förbereder talarmodeller",
                             progress=0.01)
                results = []
                for index, model_id in enumerate(request["embedding_models"]):
                    self._update(job, stage=f"Diariserar med {model_id}",
                                 progress=0.05 + 0.45 * index / max(total_models, 1))
                    try:
                        result = await runtime.diarize(
                            self.embedding_catalog[model_id], audio_path, profiles,
                            request.get("speaker_count"),
                            request.get("thresholds", {}).get(model_id),
                        )
                        result["status"] = "complete"
                    except Exception as exc:  # noqa: BLE001 - isolate model failures
                        result = {"model": model_id, "status": "failed",
                                  "error": str(exc) or type(exc).__name__, "segments": []}
                    results.append(result)
                    job["results"] = results
                    self.store.save_speaker_job(job)
                await runtime.close()

                stt_spec = self.stt_catalog[request["stt_model"]]
                successful = [item for item in results if item["status"] == "complete"]
                for index, result in enumerate(successful):
                    self._update(job, stage=f"Transkriberar {result['model']}",
                                 progress=0.52 + 0.42 * index / max(len(successful), 1))
                    started = time.perf_counter()
                    result["segments"] = await self._transcribe(
                        audio_path, result["segments"], stt_spec, request["language"])
                    transcription_seconds = time.perf_counter() - started
                    timing = result.setdefault("timing", {})
                    timing["transcription_seconds"] = transcription_seconds
                    processing = timing.get("embedding_and_clustering_seconds", 0)
                    timing["end_to_end_rtf"] = (
                        (processing + transcription_seconds) / recording["duration"]
                        if recording["duration"] else None
                    )
                    result["metrics"] = evaluate(
                        reference, result["segments"], recording["duration"])
                    job["results"] = results
                    self.store.save_speaker_job(job)

                oracle = None
                if reference and reference["segments"]:
                    self._update(job, stage="Beräknar oracle-WER", progress=0.95)
                    oracle_segments = await self._transcribe(
                        audio_path, reference["segments"], stt_spec, request["language"])
                    ref_text = " ".join(item["text"] for item in reference["segments"])
                    hyp_text = " ".join(item["text"] for item in oracle_segments)
                    oracle = {**accuracy(ref_text, hyp_text), "segments": oracle_segments}
                status = ("complete"
                          if all(item["status"] == "complete" for item in results)
                          else "complete_with_errors")
                self._update(job, status=status, stage="Klar", progress=1.0,
                             results=results, oracle=oracle,
                             completed_at=datetime.now(UTC).isoformat())
        except asyncio.CancelledError:
            self._update(job, status="cancelled", stage="Avbruten",
                         completed_at=datetime.now(UTC).isoformat())
            await self.stt_runtime.close()
        except Exception as exc:  # noqa: BLE001 - persisted job boundary
            self._update(job, status="failed", stage="Misslyckades",
                         error=str(exc) or type(exc).__name__,
                         completed_at=datetime.now(UTC).isoformat())
        finally:
            await runtime.close()
            self.tasks.pop(ident, None)

    async def cancel(self, ident):
        job = self.store.speaker_job(ident)
        if job["status"] not in {"queued", "running"}:
            return job
        self._update(job, status="cancelled", stage="Avbruten",
                     completed_at=datetime.now(UTC).isoformat())
        task = self.tasks.get(ident)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return self.store.speaker_job(ident)

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
