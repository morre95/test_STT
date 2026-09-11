import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def speaker_models():
    python = os.environ.get("STT_SPEAKER_PYTHON",
                            str(ROOT / ".venv-speaker" / "bin" / "python"))
    speaker_root = os.environ.get(
        "STT_3D_SPEAKER_ROOT", str(ROOT / "runtime-models" / "3D-Speaker"))
    specs = [
        ("campplus", "CAM++ 200k", "3D-Speaker",
         "iic/speech_campplus_sv_zh-cn_16k-common", "v1.0.0"),
        ("eres2net", "ERes2Net 200k", "3D-Speaker",
         "iic/speech_eres2net_sv_zh-cn_16k-common", "v1.0.5"),
        ("eres2netv2", "ERes2NetV2 200k", "3D-Speaker",
         "iic/speech_eres2netv2_sv_zh-cn_16k-common", "v1.0.1"),
        ("wespeaker-resnet34-lm", "WeSpeaker ResNet34-LM", "WeSpeaker",
         "voxceleb_resnet34_LM.onnx", "official-onnx"),
    ]
    return {ident: {"id": ident, "name": name, "toolkit": toolkit,
                    "checkpoint": checkpoint, "revision": revision,
                    "python": python, "sample_rate": 16000,
                    "source_path": speaker_root if toolkit == "3D-Speaker" else None,
                    "license_note": ("VoxCeleb model: dataset terms/CC BY 4.0"
                                     if toolkit == "WeSpeaker" else
                                     "3D-Speaker code: Apache-2.0; check model card terms")}
            for ident, name, toolkit, checkpoint, revision in specs}


def public_speaker_models(catalog):
    def available(spec):
        return (Path(spec["python"]).is_file() and
                (not spec.get("source_path") or Path(spec["source_path"]).is_dir()))

    return [{**{key: value for key, value in spec.items()
                if key not in {"python", "source_path"}},
             "available": available(spec),
             "availability_note": ("Runtime installed; weights are checked on first load"
                                   if available(spec)
                                   else "Install the speaker runtime (see README)")}
            for spec in catalog.values()]
