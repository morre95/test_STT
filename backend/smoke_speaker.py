"""Load one real speaker model and diarize a WAV/FLAC fixture."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from stt_lab.speaker_catalog import speaker_models
from stt_lab.speaker_runtime import SpeakerRuntime


async def run(model_id: str, audio: Path):
    catalog = speaker_models()
    if model_id not in catalog:
        raise SystemExit(f"Unknown model. Choose one of: {', '.join(catalog)}")
    if not audio.is_file():
        raise SystemExit(f"Audio file does not exist: {audio}")
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    runtime = SpeakerRuntime(data_dir)
    try:
        result = await runtime.diarize(catalog[model_id], audio, [])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=speaker_models())
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.model, args.audio))


if __name__ == "__main__":
    main()
