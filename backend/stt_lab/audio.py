"""Validation and normalization for user-owned audio files."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

MAX_AUDIO_BYTES = 200 * 1024 * 1024
MAX_AUDIO_SECONDS = 60 * 60
SAMPLE_RATE = 16_000


class AudioValidationError(ValueError):
    pass


async def receive_upload(upload, data_dir: Path) -> Path:
    """Stream an UploadFile to a private temporary path with a hard size cap."""
    suffix = Path(upload.filename or "audio").suffix.lower()
    if suffix not in {".wav", ".flac"}:
        raise AudioValidationError("Only WAV and FLAC files are supported")
    folder = data_dir / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.NamedTemporaryFile(dir=folder, suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        try:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise AudioValidationError("Audio file exceeds the 200 MiB limit")
                handle.write(chunk)
            if total == 0:
                raise AudioValidationError("Audio file is empty")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
    return path


def normalize_audio(source: Path, target: Path, *, max_seconds=MAX_AUDIO_SECONDS) -> float:
    """Decode, downmix and resample audio into canonical mono PCM16 WAV."""
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(source)
        if info.frames <= 0 or info.samplerate <= 0:
            raise AudioValidationError("Audio file contains no samples")
        duration = info.frames / info.samplerate
        if duration > max_seconds:
            raise AudioValidationError(f"Audio is longer than {max_seconds:g} seconds")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            count = max(1, round(len(mono) * SAMPLE_RATE / rate))
            old = np.arange(len(mono), dtype=np.float64)
            new = np.arange(count, dtype=np.float64) * rate / SAMPLE_RATE
            mono = np.interp(new, old, mono).astype(np.float32)
        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
        mono = np.clip(mono, -1.0, 1.0)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".normalizing.wav")
        sf.write(temporary, mono, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        temporary.replace(target)
        return len(mono) / SAMPLE_RATE
    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(f"Could not decode audio: {exc}") from exc


def copy_pcm_wav(source: Path, target: Path) -> None:
    """Small helper used when an already-normalized recording becomes a sample."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
