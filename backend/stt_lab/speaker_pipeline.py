"""Shared diarization pipeline used inside the isolated speaker environment."""
from __future__ import annotations

import importlib
import math
import time
import urllib.request
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

WESPEAKER_URL = (
    "https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/"
    "voxceleb_resnet34_LM.onnx"
)

THREED_CONFIG = {
    "campplus": {
        "object": "speakerlab.models.campplus.DTDNN.CAMPPlus",
        "args": {"feat_dim": 80, "embedding_size": 192},
        "file": "campplus_cn_common.bin",
    },
    "eres2net": {
        "object": "speakerlab.models.eres2net.ERes2Net_huge.ERes2Net",
        "args": {"feat_dim": 80, "embedding_size": 192},
        "file": "pretrained_eres2net_aug.ckpt",
    },
    "eres2netv2": {
        "object": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
        "args": {"feat_dim": 80, "embedding_size": 192,
                 "baseWidth": 26, "scale": 2, "expansion": 2},
        "file": "pretrained_eres2netv2.ckpt",
    },
}


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / max(float(norm), 1e-12)


def _load_normalized_wave(path):
    """Read the PCM16/16 kHz files produced by ``normalize_audio``.

    TorchAudio 2.9+ routes ``torchaudio.load`` through TorchCodec. The speaker
    pipeline only consumes our own normalized WAV files, so the standard
    library is both sufficient and avoids an unnecessary FFmpeg/TorchCodec
    dependency in the isolated model environment.
    """
    import torch

    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Speaker audio must be normalized mono PCM16 at 16 kHz")
        pcm = source.readframes(source.getnframes())
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples)


def _dynamic(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


class ThreeDSpeakerAdapter:
    def __init__(self, spec, cache_dir):
        import torch
        from modelscope.hub.snapshot_download import snapshot_download
        from speakerlab.process.processor import FBank

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = THREED_CONFIG[spec["id"]]
        snapshot = Path(snapshot_download(spec["checkpoint"], revision=spec["revision"],
                                          cache_dir=str(cache_dir / "modelscope")))
        state = torch.load(snapshot / config["file"], map_location="cpu")
        self.model = _dynamic(config["object"])(**config["args"])
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.frontend = FBank(80, sample_rate=16000, mean_nor=True)
        self.metadata = {"backend": "3D-Speaker/ModelScope", "device": str(self.device),
                         "embedding_size": config["args"]["embedding_size"]}

    def embed(self, waveform):
        torch = self.torch
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        waveform = waveform.reshape(1, -1).cpu()
        features = self.frontend(waveform).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = self.model(features).detach().squeeze(0).cpu().numpy()
        return _unit(result)


class WeSpeakerOnnxAdapter:
    def __init__(self, spec, cache_dir):
        import onnxruntime as ort

        folder = cache_dir / "wespeaker"
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / spec["checkpoint"]
        if not model_path.is_file():
            temporary = model_path.with_suffix(".download")
            urllib.request.urlretrieve(WESPEAKER_URL, temporary)
            temporary.replace(model_path)
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.metadata = {"backend": "WeSpeaker/ONNX Runtime",
                         "device": self.session.get_providers()[0],
                         "embedding_size": self.session.get_outputs()[0].shape[-1]}

    def embed(self, waveform):
        import torch
        from torchaudio.compliance import kaldi

        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        waveform = waveform.reshape(1, -1).cpu() * (1 << 15)
        features = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25,
                               frame_shift=10, dither=0.0, sample_frequency=16000)
        features = features - features.mean(dim=0, keepdim=True)
        shape = self.session.get_inputs()[0].shape
        value = features.unsqueeze(0).numpy().astype(np.float32)
        if len(shape) == 4:
            value = value[:, None, :, :]
        result = self.session.run([self.output_name], {self.input_name: value})[0]
        return _unit(result)


def load_adapter(spec, cache_dir):
    if spec["toolkit"] == "WeSpeaker":
        return WeSpeakerOnnxAdapter(spec, cache_dir)
    return ThreeDSpeakerAdapter(spec, cache_dir)


@lru_cache(maxsize=1)
def _vad_model():
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True)


def _speech_regions(waveform):
    from silero_vad import get_speech_timestamps

    timestamps = get_speech_timestamps(
        waveform, _vad_model(), threshold=0.5, sampling_rate=16000,
        min_speech_duration_ms=250, min_silence_duration_ms=100,
        speech_pad_ms=30, return_seconds=True,
    )
    return [(max(0.0, float(item["start"])), float(item["end"])) for item in timestamps]


def _windows(regions, total_seconds, *, width=1.5, period=0.75):
    result = []
    for start, end in regions:
        end = min(end, total_seconds)
        if end - start <= width:
            result.append((start, end))
            continue
        cursor = start
        while cursor + width < end:
            result.append((cursor, cursor + width))
            cursor += period
        final_start = max(start, end - width)
        if not result or abs(result[-1][0] - final_start) > 1e-4:
            result.append((final_start, end))
    return result


def spectral_cluster(embeddings, speaker_count=None, max_speakers=20):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from sklearn.cluster import KMeans

    values = np.asarray(embeddings, dtype=np.float32)
    count = len(values)
    if count <= 2:
        return np.zeros(count, dtype=int)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    affinity = 0.5 * (1.0 + values @ values.T)
    keep = max(2, min(50, math.ceil(count * 0.01)))
    pruned = np.zeros_like(affinity)
    indexes = np.argpartition(affinity, -keep, axis=1)[:, -keep:]
    rows = np.arange(count)[:, None]
    pruned[rows, indexes] = affinity[rows, indexes]
    affinity = 0.5 * (pruned + pruned.T)
    np.fill_diagonal(affinity, 0.0)
    laplacian = np.diag(affinity.sum(axis=1)) - affinity
    maximum = min(max_speakers, count - 1)
    wanted = int(speaker_count) if speaker_count else maximum + 1
    if count <= 500:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    else:
        wanted = max(1, min(wanted, count - 1))
        eigenvalues, eigenvectors = eigsh(
            csr_matrix(laplacian), k=wanted, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if speaker_count is None:
        candidates = min(maximum, len(eigenvalues) - 1)
        clusters = int(np.argmax(np.diff(eigenvalues[:candidates + 1])) + 1)
    else:
        clusters = max(1, min(int(speaker_count), count))
    if clusters == 1:
        return np.zeros(count, dtype=int)
    data = eigenvectors[:, :clusters]
    return KMeans(n_clusters=clusters, random_state=0, n_init=10).fit_predict(data)


def _turns(regions, windows, labels):
    turns = []
    for region_start, region_end in regions:
        members = [(window, int(label)) for window, label in zip(windows, labels)
                   if window[0] >= region_start - 1e-4 and window[1] <= region_end + 1e-4]
        if not members:
            continue
        centers = [(start + end) / 2 for (start, end), _ in members]
        for index, (_, label) in enumerate(members):
            start = region_start if index == 0 else (centers[index - 1] + centers[index]) / 2
            end = region_end if index == len(members) - 1 else (centers[index] + centers[index + 1]) / 2
            if turns and turns[-1]["cluster_id"] == label and abs(turns[-1]["end_ms"] - start * 1000) < 2:
                turns[-1]["end_ms"] = round(end * 1000)
            else:
                turns.append({"start_ms": round(start * 1000), "end_ms": round(end * 1000),
                              "cluster_id": label})
    return turns


def _profile_embedding(adapter, sample, cache_dir):
    folder = cache_dir / "profile-embeddings" / sample["model_id"]
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / f"{sample['id']}.npy"
    if cached.is_file():
        return _unit(np.load(cached))
    waveform = _load_normalized_wave(sample["path"])
    regions = _speech_regions(waveform)
    windows = _windows(regions, waveform.shape[-1] / 16000)
    embeddings = [adapter.embed(waveform[int(start * 16000):int(end * 16000)])
                  for start, end in windows]
    if not embeddings:
        raise ValueError(f"No speech found in profile sample {sample['id']}")
    result = _unit(np.mean(embeddings, axis=0))
    np.save(cached, result)
    return result


def _calibrate(profiles, vectors, manual_threshold):
    centroids = {ident: _unit(np.mean(items, axis=0))
                 for ident, items in vectors.items() if items}
    sufficient = len(profiles) >= 2 and all(len(vectors[p["id"]]) >= 2 for p in profiles)
    if not sufficient:
        if manual_threshold is not None and centroids:
            return {"available": True, "threshold": float(manual_threshold),
                    "automatic_threshold": None, "manual_override": True,
                    "reason": "Manual threshold; automatic calibration needs two profiles "
                              "with two samples each"}, centroids
        return {"available": False,
                "reason": "At least two profiles with two samples each are required"}, centroids
    genuine, impostor = [], []
    for profile in profiles:
        ident = profile["id"]
        for index, vector in enumerate(vectors[ident]):
            own = _unit(np.mean([item for offset, item in enumerate(vectors[ident])
                                 if offset != index], axis=0))
            genuine.append(float(vector @ own))
            impostor.extend(float(vector @ centroid) for other, centroid in centroids.items()
                            if other != ident)
    candidates = sorted({*genuine, *impostor})
    automatic = min(candidates, key=lambda threshold: (
        abs(sum(score < threshold for score in genuine) / len(genuine) -
            sum(score >= threshold for score in impostor) / len(impostor)),
        (sum(score < threshold for score in genuine) / len(genuine) +
         sum(score >= threshold for score in impostor) / len(impostor)) / 2,
    ))
    threshold = float(manual_threshold) if manual_threshold is not None else automatic
    far = sum(score >= threshold for score in impostor) / len(impostor)
    frr = sum(score < threshold for score in genuine) / len(genuine)
    return {"available": True, "threshold": threshold, "automatic_threshold": automatic,
            "manual_override": manual_threshold is not None, "far": far, "frr": frr}, centroids


def run_diarization(spec, audio_path, cache_dir, profiles, speaker_count=None,
                    manual_threshold=None):
    import torch

    started = time.perf_counter()
    adapter = load_adapter(spec, Path(cache_dir))
    load_seconds = time.perf_counter() - started
    waveform = _load_normalized_wave(audio_path).to(torch.float32)
    duration = waveform.shape[-1] / 16000
    regions = _speech_regions(waveform)
    windows = _windows(regions, duration)
    embedding_started = time.perf_counter()
    embeddings = [adapter.embed(waveform[int(start * 16000):int(end * 16000)])
                  for start, end in windows]
    if embeddings:
        labels = spectral_cluster(embeddings, speaker_count)
        turns = _turns(regions, windows, labels)
    else:
        labels, turns = np.array([], dtype=int), []
    vectors = {}
    for profile in profiles:
        vectors[profile["id"]] = [
            _profile_embedding(adapter, {**sample, "model_id": spec["id"]}, Path(cache_dir))
            for sample in profile["samples"]
        ]
    calibration, centroids = _calibrate(profiles, vectors, manual_threshold)
    cluster_names = {}
    if calibration["available"]:
        for cluster in sorted(set(map(int, labels))):
            centroid = _unit(np.mean([embedding for embedding, label in zip(embeddings, labels)
                                      if int(label) == cluster], axis=0))
            scores = {ident: float(centroid @ vector) for ident, vector in centroids.items()}
            matched, score = max(scores.items(), key=lambda item: item[1])
            if score >= calibration["threshold"]:
                profile = next(item for item in profiles if item["id"] == matched)
                cluster_names[cluster] = (matched, profile["name"], score)
            else:
                cluster_names[cluster] = (None, None, score)
    unknown = {}
    for turn in turns:
        profile_id, name, score = cluster_names.get(turn["cluster_id"], (None, None, None))
        if name is None:
            unknown.setdefault(turn["cluster_id"], len(unknown) + 1)
            name = f"Okänd {unknown[turn['cluster_id']]}"
        turn.update(profile_id=profile_id, speaker=name, confidence=score, text="")
    processing_seconds = time.perf_counter() - embedding_started
    return {"model": spec["id"], "runtime": adapter.metadata,
            "settings": {"vad": "silero-vad-6.2", "vad_threshold": 0.5,
                         "window_seconds": 1.5, "period_seconds": 0.75,
                         "clustering": "spectral", "speaker_count": speaker_count},
            "calibration": calibration, "segments": turns,
            "detected_speakers": len(set(map(int, labels))),
            "timing": {"load_seconds": load_seconds,
                       "embedding_and_clustering_seconds": processing_seconds,
                       "rtf": processing_seconds / duration if duration else None}}


class LiveSpeakerIdentifier:
    """Identify one live PCM window against enrolled profile centroids."""

    DEFAULT_THRESHOLD = 0.5

    def __init__(self, spec, cache_dir, profiles, manual_threshold=None):
        started = time.perf_counter()
        cache_dir = Path(cache_dir)
        self.adapter = load_adapter(spec, cache_dir)
        self.profiles = {profile["id"]: profile for profile in profiles if profile["samples"]}
        if not self.profiles:
            raise ValueError("Create a speaker profile with at least one voice sample first")
        vectors = {
            ident: [
                _profile_embedding(
                    self.adapter, {**sample, "model_id": spec["id"]}, cache_dir)
                for sample in profile["samples"]
            ]
            for ident, profile in self.profiles.items()
        }
        calibration, self.centroids = _calibrate(
            list(self.profiles.values()), vectors, manual_threshold)
        self.threshold = (float(manual_threshold) if manual_threshold is not None
                          else calibration.get("threshold", self.DEFAULT_THRESHOLD))
        self.metadata = {
            "model": spec["id"], "runtime": self.adapter.metadata,
            "calibration": calibration, "threshold": self.threshold,
            "profiles": [{"id": profile["id"], "name": profile["name"]}
                         for profile in self.profiles.values()],
            "load_seconds": time.perf_counter() - started,
        }

    def identify(self, pcm):
        import torch

        started = time.perf_counter()
        waveform = torch.from_numpy(
            np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
        regions = _speech_regions(waveform)
        speech_seconds = sum(end - start for start, end in regions)
        if speech_seconds < 0.25:
            return {"speech": False, "speaker": None, "profile_id": None,
                    "score": None, "scores": [],
                    "processing_ms": (time.perf_counter() - started) * 1000}

        embedding = self.adapter.embed(waveform)
        scores = sorted(({
            "profile_id": ident,
            "speaker": self.profiles[ident]["name"],
            "score": float(embedding @ centroid),
        } for ident, centroid in self.centroids.items()),
            key=lambda item: item["score"], reverse=True)
        best = scores[0]
        known = best["score"] >= self.threshold
        return {"speech": True,
                "speaker": best["speaker"] if known else "Okänd",
                "profile_id": best["profile_id"] if known else None,
                "score": best["score"], "scores": scores,
                "processing_ms": (time.perf_counter() - started) * 1000}
