import asyncio
import contextlib
import os
import time
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .audio import AudioValidationError, normalize_audio, receive_upload
from .catalog import models, public_models
from .runtime import Runtime
from .speaker_catalog import public_speaker_models, speaker_models
from .speaker_jobs import SpeakerJobs
from .speaker_runtime import LiveSpeakerRuntime
from .store import Store

BYTES_PER_SECOND = 32000
FRAME_BYTES = 3200
MAX_SECONDS = 3600
LIVE_SPEAKER_WINDOW_BYTES = round(1.5 * BYTES_PER_SECOND)
LIVE_SPEAKER_STEP_BYTES = round(0.75 * BYTES_PER_SECOND)


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    model: str
    language: Literal["sv", "en", "auto"] = "sv"
    recording_id: str | None = None
    settings: dict = Field(default_factory=dict)


class Reference(BaseModel):
    text: str = Field(max_length=50000)


class ClientMetrics(BaseModel):
    first_text_ms: float | None = Field(default=None, ge=0, le=600000)
    finalization_ms: float | None = Field(default=None, ge=0, le=600000)


class ReferenceSpeaker(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    profile_id: str | None = None


class ReferenceSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker_id: str
    text: str = Field(min_length=1, max_length=10000)


class SpeakerReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speakers: list[ReferenceSpeaker] = Field(min_length=1, max_length=100)
    segments: list[ReferenceSegment] = Field(min_length=1, max_length=10000)


class ProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)


class SpeakerBenchmarkStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recording_id: str
    embedding_models: list[str] = Field(min_length=1, max_length=20)
    stt_model: str
    language: Literal["sv", "en", "auto"] = "sv"
    speaker_count: int | None = Field(default=None, ge=1, le=20)
    thresholds: dict[str, float] = Field(default_factory=dict)


class LiveSpeakerStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    model: str
    threshold: float | None = Field(default=None, ge=-1, le=1)


# Vosk decodes every frame it is given, so it exposes no streaming tunables.
SETTING_CHOICES = {
    "qwen": {"chunk_seconds": (0.5, 1, 2, 3)},
    "nemotron": {"chunk_ms": (80, 160, 320, 560, 1120)},
    "vosk": {},
}


def validated_settings(spec, settings):
    settings = {**spec["defaults"], **settings}
    choices = SETTING_CHOICES[spec["family"]]
    if set(settings) != set(choices):
        raise ValueError(f"{spec['family']} accepts exactly these settings: "
                         f"{', '.join(choices) or 'none'}")
    for key, allowed in choices.items():
        if settings[key] not in allowed:
            raise ValueError(f"{key} must be one of "
                             f"{', '.join(str(value) for value in allowed)}")
    return settings


def create_app(data_dir=None, runtime=None, catalog=None, embedding_catalog=None,
               speaker_runtime_factory=None, live_speaker_runtime_factory=None):
    data_dir = Path(data_dir or os.environ.get("STT_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    store = Store(data_dir)
    engine = runtime or Runtime(data_dir)
    catalog = catalog or models()
    embedding_catalog = embedding_catalog or speaker_models()
    busy = asyncio.Lock()
    jobs = SpeakerJobs(store, embedding_catalog, catalog, engine, busy,
                       **({"runtime_factory": speaker_runtime_factory}
                          if speaker_runtime_factory else {}))
    live_runtime_factory = live_speaker_runtime_factory or LiveSpeakerRuntime

    @asynccontextmanager
    async def lifespan(app):
        yield
        await jobs.close()
        await engine.close()

    app = FastAPI(title="STT Lab", lifespan=lifespan)
    app.state.store = store
    app.state.speaker_jobs = jobs

    @app.get("/health")
    async def health():
        return {"status": "ok", "busy": busy.locked(), "active_model": engine.model_id,
                "active_speaker_jobs": len(jobs.tasks)}

    @app.get("/models")
    async def model_list():
        return public_models(catalog)

    @app.get("/speaker-models")
    async def speaker_model_list():
        return public_speaker_models(embedding_catalog)

    @app.get("/recordings")
    async def recordings():
        return store.history()

    @app.post("/recordings/upload", status_code=201)
    async def upload_recording(file: Annotated[UploadFile, File()]):
        temporary = normalized = None
        try:
            temporary = await receive_upload(file, data_dir)
            normalized = temporary.with_suffix(".normalized.wav")
            duration = normalize_audio(temporary, normalized)
            ident = store.create_recording(source="upload", original_name=file.filename)
            normalized.replace(store.audio_path(ident))
            store.finish_recording(ident, duration, True)
            return store.recording(ident)
        except AudioValidationError as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
            if normalized:
                normalized.unlink(missing_ok=True)

    @app.get("/recordings/{ident}/audio")
    async def audio(ident: str):
        try:
            path = store.audio_path(ident)
            if not path.is_file():
                raise KeyError(ident)
            return FileResponse(path, media_type="audio/wav", filename=f"{ident}.wav")
        except KeyError:
            raise HTTPException(404, "Recording not found") from None

    @app.put("/recordings/{ident}/reference")
    async def reference(ident: str, body: Reference):
        try:
            store.set_reference(ident, body.text)
        except KeyError:
            raise HTTPException(404, "Recording not found") from None
        return {"ok": True}

    @app.put("/recordings/{ident}/speaker-reference")
    async def speaker_reference(ident: str, body: SpeakerReference):
        try:
            recording = store.recording(ident)
        except KeyError:
            raise HTTPException(404, "Recording not found") from None
        speaker_ids = [item.id for item in body.speakers]
        if len(set(speaker_ids)) != len(speaker_ids):
            raise HTTPException(422, "Speaker ids must be unique")
        known_profiles = {item["id"] for item in store.profiles()}
        for speaker in body.speakers:
            if speaker.profile_id and speaker.profile_id not in known_profiles:
                raise HTTPException(422, f"Unknown profile: {speaker.profile_id}")
        duration_ms = round(recording["duration"] * 1000)
        by_speaker = {}
        for segment in sorted(body.segments, key=lambda item: (item.speaker_id, item.start_ms)):
            if segment.speaker_id not in speaker_ids:
                raise HTTPException(422, f"Unknown reference speaker: {segment.speaker_id}")
            if segment.start_ms >= segment.end_ms or segment.end_ms > duration_ms:
                raise HTTPException(422, "Reference segment is outside the recording")
            previous_end = by_speaker.get(segment.speaker_id, -1)
            if segment.start_ms < previous_end:
                raise HTTPException(422, "One speaker cannot have overlapping segments")
            by_speaker[segment.speaker_id] = segment.end_ms
        data = body.model_dump()
        store.set_speaker_reference(ident, data)
        return data

    @app.get("/speaker-profiles")
    async def profile_list():
        return store.profiles()

    @app.post("/speaker-profiles", status_code=201)
    async def create_profile(body: ProfileCreate):
        name = " ".join(body.name.split())
        if any(item["name"].casefold() == name.casefold() for item in store.profiles()):
            raise HTTPException(409, "A speaker profile with that name already exists")
        return store.create_profile(name)

    @app.get("/speaker-profiles/{ident}")
    async def get_profile(ident: str):
        try:
            return store.profile(ident)
        except KeyError:
            raise HTTPException(404, "Speaker profile not found") from None

    @app.put("/speaker-profiles/{ident}")
    async def update_profile(ident: str, body: ProfileCreate):
        name = " ".join(body.name.split())
        if any(item["id"] != ident and item["name"].casefold() == name.casefold()
               for item in store.profiles()):
            raise HTTPException(409, "A speaker profile with that name already exists")
        try:
            return store.update_profile(ident, name)
        except KeyError:
            raise HTTPException(404, "Speaker profile not found") from None

    @app.delete("/speaker-profiles/{ident}")
    async def delete_profile(ident: str):
        try:
            profile = store.profile(ident)
            paths = [store.profile_sample_path(ident, item["id"])
                     for item in profile["samples"]]
            store.delete_profile(ident)
        except KeyError:
            raise HTTPException(404, "Speaker profile not found") from None
        for path in paths:
            path.unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/speaker-profiles/{ident}/samples", status_code=201)
    async def add_profile_sample(ident: str, file: Annotated[UploadFile, File()]):
        try:
            store.profile(ident)
        except KeyError:
            raise HTTPException(404, "Speaker profile not found") from None
        temporary = normalized = None
        try:
            temporary = await receive_upload(file, data_dir)
            normalized = temporary.with_suffix(".normalized.wav")
            duration = normalize_audio(temporary, normalized, max_seconds=60)
            if duration < 2:
                raise AudioValidationError("A profile sample must be at least two seconds")
            sample = store.add_profile_sample(ident, duration, file.filename)
            normalized.replace(store.profile_sample_path(ident, sample["id"]))
            return sample
        except AudioValidationError as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
            if normalized:
                normalized.unlink(missing_ok=True)

    @app.delete("/speaker-profiles/{profile_id}/samples/{sample_id}")
    async def delete_profile_sample(profile_id: str, sample_id: str):
        try:
            path = store.profile_sample_path(profile_id, sample_id)
            store.delete_profile_sample(sample_id)
        except KeyError:
            raise HTTPException(404, "Speaker sample not found") from None
        path.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/speaker-benchmarks")
    async def speaker_benchmark_list():
        return store.speaker_jobs()

    @app.post("/speaker-benchmarks", status_code=202)
    async def start_speaker_benchmark(body: SpeakerBenchmarkStart):
        try:
            recording = store.recording(body.recording_id)
        except KeyError:
            raise HTTPException(404, "Recording not found") from None
        if not recording["complete"] or recording["duration"] <= 0:
            raise HTTPException(409, "Recording is incomplete or empty")
        if len(set(body.embedding_models)) != len(body.embedding_models):
            raise HTTPException(422, "Embedding models must be unique")
        unknown = set(body.embedding_models) - set(embedding_catalog)
        if unknown:
            raise HTTPException(422, f"Unknown embedding models: {', '.join(sorted(unknown))}")
        if body.stt_model not in catalog:
            raise HTTPException(422, "Unknown STT model")
        if body.language not in catalog[body.stt_model]["languages"]:
            raise HTTPException(422, "The STT model does not support that language")
        if set(body.thresholds) - set(body.embedding_models):
            raise HTTPException(422, "Thresholds may only target selected embedding models")
        if any(not -1 <= value <= 1 for value in body.thresholds.values()):
            raise HTTPException(422, "Cosine thresholds must be between -1 and 1")
        unavailable = [
            ident for ident in body.embedding_models
            if not Path(embedding_catalog[ident]["python"]).is_file()
            or (embedding_catalog[ident].get("source_path")
                and not Path(embedding_catalog[ident]["source_path"]).is_dir())
        ]
        if unavailable:
            raise HTTPException(409, f"Speaker runtime is not installed: {', '.join(unavailable)}")
        if not Path(catalog[body.stt_model]["python"]).is_file():
            raise HTTPException(409, "Selected STT runtime is not installed")
        return jobs.submit(body.model_dump())

    @app.get("/speaker-benchmarks/{ident}")
    async def speaker_benchmark(ident: str):
        try:
            return store.speaker_job(ident)
        except KeyError:
            raise HTTPException(404, "Speaker benchmark not found") from None

    @app.post("/speaker-benchmarks/{ident}/cancel")
    async def cancel_speaker_benchmark(ident: str):
        try:
            return await jobs.cancel(ident)
        except KeyError:
            raise HTTPException(404, "Speaker benchmark not found") from None

    @app.patch("/runs/{ident}/client-metrics")
    async def client_metrics(ident: str, body: ClientMetrics):
        try:
            run = store.run(ident)
        except KeyError:
            raise HTTPException(404, "Run not found") from None
        if run["status"] != "complete":
            raise HTTPException(409, "Run is not complete")
        run["client_metrics"] = body.model_dump()
        store.save_run(run)
        return {"ok": True}

    @app.websocket("/ws/speaker-identify")
    async def identify_speaker(ws: WebSocket):
        await ws.accept()
        if busy.locked():
            await ws.send_json({"type": "error", "message": "Another test is running"})
            await ws.close(code=1013)
            return
        await busy.acquire()
        speaker_runtime = live_runtime_factory(data_dir)
        try:
            start = LiveSpeakerStart.model_validate(
                await asyncio.wait_for(ws.receive_json(), 15))
            if start.model not in embedding_catalog:
                raise ValueError("Unknown speaker model")
            spec = embedding_catalog[start.model]
            if not Path(spec["python"]).is_file():
                raise ValueError("Speaker runtime is not installed")
            if spec.get("source_path") and not Path(spec["source_path"]).is_dir():
                raise ValueError("3D-Speaker source is not installed")
            profiles = [item for item in jobs.profiles() if item["samples"]]
            if not profiles:
                raise ValueError("Create a speaker profile with at least one voice sample first")

            await ws.send_json({"type": "loading", "model": start.model})
            metadata = await speaker_runtime.load(spec, profiles, start.threshold)
            await ws.send_json({"type": "ready", **metadata})
            window = bytearray()
            since_result = 0
            while True:
                message = await asyncio.wait_for(ws.receive(), 30)
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if message.get("bytes") is not None:
                    pcm = message["bytes"]
                    if not pcm or len(pcm) > FRAME_BYTES or len(pcm) % 2:
                        raise ValueError("Audio frames must contain 1–1600 PCM16 samples")
                    window.extend(pcm)
                    since_result += len(pcm)
                    if len(window) > LIVE_SPEAKER_WINDOW_BYTES:
                        del window[:-LIVE_SPEAKER_WINDOW_BYTES]
                    if (len(window) == LIVE_SPEAKER_WINDOW_BYTES and
                            since_result >= LIVE_SPEAKER_STEP_BYTES):
                        since_result = 0
                        result = await speaker_runtime.identify(bytes(window))
                        await ws.send_json({"type": "speaker", **result})
                else:
                    import json
                    control = json.loads(message.get("text", ""))
                    if control != {"type": "stop"}:
                        raise ValueError("Expected PCM audio or stop")
                    await ws.send_json({"type": "stopped"})
                    return
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - WebSocket protocol boundary
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "error",
                                    "message": str(exc) or type(exc).__name__})
        finally:
            await speaker_runtime.close()
            busy.release()
            with contextlib.suppress(Exception):
                await ws.close()

    @app.websocket("/ws/transcribe")
    async def transcribe(ws: WebSocket):
        await ws.accept()
        if busy.locked():
            await ws.send_json({"type": "error", "message": "Another test is running"})
            await ws.close(code=1013)
            return
        await busy.acquire()
        run = None
        writer = None
        tasks = []
        received = 0
        replay = False
        input_complete = False
        sequence = 0
        processing = 0.0
        try:
            start = Start.model_validate(await asyncio.wait_for(ws.receive_json(), 15))
            if start.model not in catalog:
                raise ValueError("Unknown model")
            spec = catalog[start.model]
            if start.language not in spec["languages"]:
                raise ValueError(f"{spec['name']} only supports "
                                 f"{', '.join(spec['languages'])}")
            settings = validated_settings(spec, start.settings)
            replay = start.recording_id is not None
            if replay:
                recording = store.recording(start.recording_id)
                if not recording["complete"] or recording["duration"] <= 0:
                    raise ValueError("Only complete, nonempty recordings can be replayed")
                recording_id = start.recording_id
            else:
                recording_id = store.create_recording()
            run = {"id": str(uuid4()), "recording_id": recording_id,
                   "created_at": datetime.now(UTC).isoformat(),
                   "model": start.model, "checkpoint": spec["checkpoint"],
                   "language": start.language, "settings": settings,
                   "mode": "replay" if replay else "live", "status": "running", "text": ""}
            store.save_run(run)
            await ws.send_json({"type": "loading", "run_id": run["id"], "recording_id": recording_id})
            load_started = time.perf_counter()
            run["runtime"] = await engine.load(spec)
            await engine.start(start.language, settings)
            run["load_ms"] = (time.perf_counter() - load_started) * 1000
            if not replay:
                writer = wave.open(str(store.audio_path(recording_id)), "wb")  # noqa: SIM115
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
            await ws.send_json({"type": "ready", "run_id": run["id"],
                                "recording_id": recording_id, "load_ms": run["load_ms"]})
            queue = asyncio.Queue(maxsize=100)
            origin = None
            stop_time = None

            def enqueue(pcm):
                nonlocal received, origin
                if not pcm or len(pcm) > FRAME_BYTES or len(pcm) % 2:
                    raise ValueError("Audio frames must contain 1–1600 PCM16 samples")
                if received + len(pcm) > MAX_SECONDS * BYTES_PER_SECOND:
                    raise ValueError("Maximum recording length is five minutes")
                if origin is None:
                    origin = time.perf_counter()
                received += len(pcm)
                if writer:
                    writer.writeframesraw(pcm)
                try:
                    queue.put_nowait(pcm)
                except asyncio.QueueFull:
                    raise ValueError("Model cannot keep up: ten-second audio queue is full") from None

            async def receive_live():
                nonlocal input_complete, stop_time
                while True:
                    message = await asyncio.wait_for(ws.receive(), 30)
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    if message.get("bytes") is not None:
                        enqueue(message["bytes"])
                    else:
                        import json
                        control = json.loads(message.get("text", ""))
                        if control != {"type": "stop"}:
                            raise ValueError("Expected PCM audio or stop")
                        input_complete = True
                        stop_time = time.perf_counter()
                        await queue.put(None)
                        return

            async def feed_replay():
                nonlocal input_complete, stop_time
                started = time.perf_counter()
                with wave.open(str(store.audio_path(recording_id)), "rb") as source:
                    while pcm := source.readframes(FRAME_BYTES // 2):
                        # Release only audio whose capture time has elapsed.
                        due = started + (received + len(pcm)) / BYTES_PER_SECOND
                        await asyncio.sleep(max(0, due - time.perf_counter()))
                        enqueue(pcm)
                input_complete = True
                stop_time = time.perf_counter()
                await queue.put(None)

            async def watch_replay():
                await ws.receive()
                raise ValueError("Replay cancelled or disconnected")

            async def consume():
                nonlocal processing, sequence
                last_text = ""
                while True:
                    pcm = await queue.get()
                    before = time.perf_counter()
                    result = await (engine.finish() if pcm is None else engine.push(pcm))
                    processing += result.get("processing_ms", (time.perf_counter() - before) * 1000)
                    text = result.get("text", "")
                    run["text"] = text
                    if text and "server_first_text_ms" not in run and origin is not None:
                        run["server_first_text_ms"] = (time.perf_counter() - origin) * 1000
                    if pcm is None:
                        run["server_finalization_ms"] = (time.perf_counter() - stop_time) * 1000
                        run["detected_language"] = result.get("language")
                        return
                    if text != last_text:
                        sequence += 1
                        await ws.send_json({"type": "partial", "run_id": run["id"],
                                            "seq": sequence, "text": text})
                        last_text = text

            producer = asyncio.create_task(feed_replay() if replay else receive_live())
            consumer = asyncio.create_task(consume())
            tasks = [producer, consumer]
            if replay:
                tasks.append(asyncio.create_task(watch_replay()))
            # Propagate producer failures immediately even while inference is busy.
            pending = set(tasks)
            while consumer in pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            duration = received / BYTES_PER_SECOND
            run.update(status="complete", duration=duration, processing_ms=processing,
                       rtf=processing / 1000 / duration if duration else None)
            if writer:
                writer.close()
                writer = None
                store.finish_recording(recording_id, duration, True)
            store.save_run(run)
            await ws.send_json({"type": "final", "seq": sequence + 1, **run})
        except Exception as exc:  # noqa: BLE001 - WebSocket protocol boundary
            if run:
                run.update(status="failed", error=str(exc) or type(exc).__name__,
                           duration=received / BYTES_PER_SECOND, processing_ms=processing)
                store.save_run(run)
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "error", "message": str(exc) or type(exc).__name__,
                                    "run_id": run["id"] if run else None})
            await engine.close()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if writer:
                writer.close()
                store.finish_recording(run["recording_id"], received / BYTES_PER_SECOND,
                                       input_complete)
            busy.release()
            with contextlib.suppress(Exception):
                await ws.close()

    return app
