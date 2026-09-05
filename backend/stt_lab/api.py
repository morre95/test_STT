import asyncio
import contextlib
import os
import time
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .catalog import models, public_models
from .runtime import Runtime
from .store import Store

BYTES_PER_SECOND = 32000
FRAME_BYTES = 3200
MAX_SECONDS = 300


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


def validated_settings(spec, settings):
    settings = {**spec["defaults"], **settings}
    if spec["family"] == "qwen":
        if set(settings) != {"chunk_seconds"} or settings["chunk_seconds"] not in (0.5, 1, 2, 3):
            raise ValueError("Qwen chunk_seconds must be 0.5, 1, 2 or 3")
    elif set(settings) != {"chunk_ms"} or settings["chunk_ms"] not in (80, 160, 320, 560, 1120):
        raise ValueError("Nemotron chunk_ms must be 80, 160, 320, 560 or 1120")
    return settings


def create_app(data_dir=None, runtime=None, catalog=None):
    data_dir = Path(data_dir or os.environ.get("STT_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    store = Store(data_dir)
    engine = runtime or Runtime(data_dir)
    catalog = catalog or models()
    busy = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await engine.close()

    app = FastAPI(title="STT Lab", lifespan=lifespan)
    app.state.store = store

    @app.get("/health")
    async def health():
        return {"status": "ok", "busy": busy.locked(), "active_model": engine.model_id}

    @app.get("/models")
    async def model_list():
        return public_models(catalog)

    @app.get("/recordings")
    async def recordings():
        return store.history()

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
