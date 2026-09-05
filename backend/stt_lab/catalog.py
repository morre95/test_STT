import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def models():
    specs = [
        ("qwen-0.6b", "Qwen3-ASR 0.6B", "qwen", "Qwen/Qwen3-ASR-0.6B"),
        ("qwen-1.7b", "Qwen3-ASR 1.7B", "qwen", "Qwen/Qwen3-ASR-1.7B"),
        ("nemotron-0.6b", "Nemotron 3.5 ASR 0.6B", "nemotron",
         "nvidia/nemotron-3.5-asr-streaming-0.6b"),
    ]
    return {key: {
        "id": key, "name": name, "family": family, "checkpoint": checkpoint,
        "provider": "local", "streaming": True, "languages": ["sv", "en", "auto"],
        "python": os.environ.get(f"STT_{family.upper()}_PYTHON",
                                  str(ROOT / f".venv-{family}" / "bin/python")),
        "defaults": {"chunk_seconds": 2.0} if family == "qwen" else {"chunk_ms": 320},
    } for key, name, family, checkpoint in specs}


def public_models(catalog):
    return [{**{k: v for k, v in item.items() if k != "python"},
             "available": Path(item["python"]).is_file(),
             "availability_note": "Runtime installed; GPU and model checked on load"
             if Path(item["python"]).is_file() else "Model runtime is not installed"}
            for item in catalog.values()]
