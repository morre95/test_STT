"""JSON-lines entry point for heavyweight speaker dependencies."""
from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback


def reply(payload):
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main():
    for raw in sys.stdin:
        started = time.perf_counter()
        try:
            request = json.loads(raw)
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] != "diarize":
                    raise ValueError(f"Unknown operation: {request['op']}")
                from stt_lab.speaker_pipeline import run_diarization

                result = run_diarization(
                    request["model"], request["audio_path"], request["cache_dir"],
                    request.get("profiles", []), request.get("speaker_count"),
                    request.get("manual_threshold"),
                )
            result["worker_seconds"] = time.perf_counter() - started
            reply(result)
        except Exception as exc:  # noqa: BLE001 - worker protocol boundary
            traceback.print_exc(file=sys.stderr)
            reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
