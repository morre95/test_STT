import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILY_DEFAULTS = {
    "qwen": {"chunk_seconds": 2.0},
    "nemotron": {"chunk_ms": 320},
    "vosk": {},
}


def models():
    specs = [
        ("qwen-0.6b", "Qwen3-ASR 0.6B", "qwen", "Qwen/Qwen3-ASR-0.6B",
         ["sv", "en", "auto"]),
        ("qwen-1.7b", "Qwen3-ASR 1.7B", "qwen", "Qwen/Qwen3-ASR-1.7B",
         ["sv", "en", "auto"]),
        ("nemotron-0.6b", "Nemotron 3.5 ASR 0.6B", "nemotron",
         "nvidia/nemotron-3.5-asr-streaming-0.6b", ["sv", "en", "auto"]),
        # Vosk ships one Kaldi graph per language, so each checkpoint is single-language.
        ("vosk-sv", "Vosk Small Svenska 0.15", "vosk",
         "vosk-model-small-sv-rhasspy-0.15", ["sv"]),
        ("vosk-en", "Vosk Small English 0.15", "vosk",
         "vosk-model-small-en-us-0.15", ["en"]),
    ]
    return {key: {
        "id": key, "name": name, "family": family, "checkpoint": checkpoint,
        "provider": "local", "streaming": True, "languages": languages,
        "python": os.environ.get(f"STT_{family.upper()}_PYTHON",
                                  str(ROOT / f".venv-{family}" / "bin/python")),
        "defaults": dict(FAMILY_DEFAULTS[family]),
    } for key, name, family, checkpoint, languages in specs}


def public_models(catalog):
    return [{**{k: v for k, v in item.items() if k != "python"},
             "available": Path(item["python"]).is_file(),
             "availability_note": "Runtime installed; model and hardware checked on load"
             if Path(item["python"]).is_file() else "Model runtime is not installed"}
            for item in catalog.values()]
