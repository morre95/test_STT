"""Short-lived isolated runtime for one speaker embedding model at a time."""
from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path


class SpeakerRuntime:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.process = None
        self.log = None

    async def diarize(self, spec, audio_path, profiles, speaker_count=None, threshold=None):
        await self.close()
        python = Path(spec["python"])
        if not python.is_file():
            raise RuntimeError("Install the speaker runtime first (see README)")
        self.log = (self.log_dir / "speaker-runtime.log").open("a")
        python_paths = [str(Path(__file__).resolve().parents[1])]
        if spec.get("source_path"):
            python_paths.append(spec["source_path"])
        if os.environ.get("PYTHONPATH"):
            python_paths.append(os.environ["PYTHONPATH"])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_paths)}
        self.process = await asyncio.create_subprocess_exec(
            str(python), "-u", "-m", "stt_lab.speaker_worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self.log, start_new_session=True, env=env, limit=4 * 1024 * 1024,
        )
        message = {"op": "diarize", "model": {k: v for k, v in spec.items() if k != "python"},
                   "audio_path": str(audio_path), "cache_dir": str(self.log_dir / "models"),
                   "profiles": profiles, "speaker_count": speaker_count,
                   "manual_threshold": threshold}
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), 60 * 60)
        except TimeoutError:
            raise TimeoutError("Speaker model did not finish within one hour") from None
        while line:
            try:
                result = json.loads(line)
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.log.write(line.decode(errors="replace"))
                self.log.flush()
                line = await self.process.stdout.readline()
        else:
            raise RuntimeError("Speaker worker exited; see speaker-runtime.log")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    async def close(self):
        if self.process and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()
        if self.log:
            self.log.close()
        self.process = self.log = None
