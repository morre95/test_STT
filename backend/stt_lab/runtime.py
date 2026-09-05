"""A single active subprocess; model dependencies never enter the API process."""
import asyncio
import base64
import json
import os
import signal
from pathlib import Path


class Runtime:
    def __init__(self, log_dir: Path):
        self.process = None
        self.model_id = None
        self.log_dir = log_dir
        self.log = None
        self.metadata = {}

    async def request(self, message, timeout=120):
        if self.process is None or self.process.returncode is not None:
            raise RuntimeError("Model process is not running; see backend/data/runtime.log")
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Model process did not return a protocol response")
            line = await asyncio.wait_for(self.process.stdout.readline(), remaining)
            if not line:
                raise RuntimeError("Model process exited; see backend/data/runtime.log")
            try:
                reply = json.loads(line)
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                # vLLM child processes can inherit stdout and emit startup logs
                # despite Python-level redirection in the worker.
                if self.log:
                    self.log.write(line.decode(errors="replace"))
                    self.log.flush()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    async def load(self, spec):
        if self.model_id == spec["id"] and self.process and self.process.returncode is None:
            return self.metadata
        await self.close()
        if not Path(spec["python"]).is_file():
            raise RuntimeError(f"Install the {spec['family']} runtime first (see README)")
        self.log = (self.log_dir / "runtime.log").open("a")
        self.process = await asyncio.create_subprocess_exec(
            spec["python"], "-u", str(Path(__file__).with_name("worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self.log, start_new_session=True, limit=4 * 1024 * 1024,
        )
        self.metadata = await self.request({"op": "load", "model": spec}, timeout=1800)
        self.model_id = spec["id"]
        return self.metadata

    async def start(self, language, settings):
        return await self.request({"op": "start", "language": language, "settings": settings})

    async def push(self, pcm):
        return await self.request({"op": "audio", "pcm": base64.b64encode(pcm).decode()})

    async def finish(self):
        return await self.request({"op": "finish"})

    async def close(self):
        if self.process:
            # vLLM can have its own children: terminate the whole owned process group.
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
        self.process = self.model_id = self.log = None
        self.metadata = {}
