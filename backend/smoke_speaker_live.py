"""Load one speaker model and identify a short window against one profile."""
from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

from stt_lab.speaker_catalog import speaker_models
from stt_lab.speaker_runtime import LiveSpeakerRuntime


async def run(model_id: str, profile_audio: Path, test_audio: Path, start: float):
    spec = speaker_models()[model_id]
    data_dir = Path(__file__).resolve().parent / "data"
    profiles = [{
        "id": "smoke-profile", "name": "Referens",
        "samples": [{"id": "smoke-sample", "path": str(profile_audio.resolve())}],
    }]
    with wave.open(str(test_audio), "rb") as source:
        source.setpos(min(round(start * 16000), source.getnframes()))
        pcm = source.readframes(round(1.5 * 16000))
    runtime = LiveSpeakerRuntime(data_dir)
    try:
        metadata = await runtime.load(spec, profiles)
        result = await runtime.identify(pcm)
        print(json.dumps({"metadata": metadata, "result": result},
                         ensure_ascii=False, indent=2))
    finally:
        await runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=speaker_models())
    parser.add_argument("profile_audio", type=Path)
    parser.add_argument("test_audio", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    args = parser.parse_args()
    asyncio.run(run(args.model, args.profile_audio, args.test_audio, args.start))


if __name__ == "__main__":
    main()
