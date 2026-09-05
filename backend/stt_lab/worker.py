"""JSON-lines model worker, run in a family-specific Python environment."""
from __future__ import annotations

import base64
import contextlib
import copy
import json
import re
import sys
import time
import traceback
from typing import Any, ClassVar

import numpy as np

SAMPLE_RATE = 16000


def reply(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def to_float32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def language(code: str, family: str) -> str | None:
    if family == "qwen":
        return {"sv": "Swedish", "en": "English", "auto": None}[code]
    return {"sv": "sv-SE", "en": "en-US", "auto": "auto"}[code]


class QwenWorker:
    def __init__(self, checkpoint: str):
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.LLM(
            model=checkpoint, gpu_memory_utilization=0.82,
            max_inference_batch_size=1, max_new_tokens=256,
        )
        self.state = None

    def metadata(self):
        import qwen_asr
        return {"backend": "qwen-asr/vLLM",
                "qwen_asr": getattr(qwen_asr, "__version__", "unknown")}

    def start(self, code, settings):
        self.state = self.model.init_streaming_state(
            language=language(code, "qwen"), context="",
            unfixed_chunk_num=2, unfixed_token_num=5,
            chunk_size_sec=float(settings["chunk_seconds"]),
        )

    def audio(self, pcm):
        if self.state is None:
            raise RuntimeError("No active session")
        self.model.streaming_transcribe(to_float32(pcm), self.state)
        return {"text": self.state.text or "", "language": self.state.language}

    def finish(self):
        if self.state is None:
            raise RuntimeError("No active session")
        self.model.finish_streaming_transcribe(self.state)
        result = {"text": self.state.text or "", "language": self.state.language}
        self.state = None
        return result


class NemotronWorker:
    """Cache-aware streaming inference.

    Two parts of NeMo's streaming contract have to be honoured exactly or the
    transcript degrades into cross-language gibberish: features must come from a
    *continuous* audio stream, and the encoder must be fed the chunk, shift and
    pre-encode-cache widths from ``streaming_cfg`` — the first step uses its own,
    narrower set.
    """

    RIGHT_CONTEXT: ClassVar = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}
    # A 25 ms analysis window centred on a frame reaches 1.25 hops back, so two
    # hops of replayed audio make every emitted frame identical to the same frame
    # computed over the whole recording at once.
    LOOK_BACK_HOPS: ClassVar = 2
    # The model emits an utterance-final language token, e.g. "hej där <sv-SE>".
    LANGUAGE_TAG: ClassVar = re.compile(r"<[a-z]{2}(?:-[A-Za-z]{2,4})?>")

    def __init__(self, checkpoint: str):
        import nemo.collections.asr as nemo_asr
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Nemotron requires a working CUDA device")
        self.torch = torch
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=checkpoint)
        self.model.to(torch.device("cuda:0")).eval()
        self.pending = np.empty(0, dtype=np.float32)

    def metadata(self):
        import nemo
        return {"backend": "NVIDIA NeMo cache-aware RNNT",
                "nemo": getattr(nemo, "__version__", "unknown"),
                "cuda": self.torch.version.cuda,
                "gpu": self.torch.cuda.get_device_name(0)}

    def _make_preprocessor(self):
        from nemo.collections.asr.models import EncDecCTCModelBPE
        from omegaconf import OmegaConf

        cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        return EncDecCTCModelBPE.from_config_dict(cfg.preprocessor).to(self.model.device)

    @staticmethod
    def _sizes(value):
        """Streaming widths are either one number or [first step, steady state]."""
        return (value, value) if isinstance(value, int) else (value[0], value[1])

    def start(self, code, settings):
        chunk_ms = int(settings["chunk_ms"])
        left = self.model.encoder.att_context_size[0]
        # Also recomputes streaming_cfg, so read the widths afterwards.
        self.model.encoder.set_default_att_context_size(
            [left, self.RIGHT_CONTEXT[chunk_ms]])
        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(language(code, "nemotron"))
        self.preprocessor = self._make_preprocessor()
        streaming = self.model.encoder.streaming_cfg
        self.chunk = self._sizes(streaming.chunk_size)
        self.shift = self._sizes(streaming.shift_size)
        self.pre_encode = self._sizes(streaming.pre_encode_cache_size)
        self.minimum = self._sizes(self.model.encoder.pre_encode.get_sampling_frames())
        self.hop = round(self.model.cfg.preprocessor.window_stride * SAMPLE_RATE)
        self.look_back = np.zeros(self.LOOK_BACK_HOPS * self.hop, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.features = self.torch.zeros(
            (1, self.model.cfg.preprocessor.features, 0), device=self.model.device)
        self.cursor = self.steps = 0
        self.cache_channel, self.cache_time, self.cache_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1))
        self.hypotheses = self.prediction = None
        self.text = ""
        self.detected = None

    def _extend_features(self):
        """Turn whole hops of buffered audio into mel frames, with look-back."""
        torch = self.torch
        count = len(self.pending) // self.hop
        if not count:
            return
        samples, self.pending = self.pending[:count * self.hop], self.pending[count * self.hop:]
        segment = np.concatenate((self.look_back, samples))
        self.look_back = segment[-self.LOOK_BACK_HOPS * self.hop:]
        signal = torch.from_numpy(segment).unsqueeze(0).to(self.model.device)
        length = torch.tensor([len(segment)], device=self.model.device)
        with torch.inference_mode():
            processed, _ = self.preprocessor(input_signal=signal, length=length)
        # Frame i of the segment is centred on hop i, so our own frames start
        # where the look-back ends; the trailing frame is half zero-padding.
        start = self.LOOK_BACK_HOPS
        self.features = torch.cat([self.features, processed[:, :, start:start + count]], dim=-1)

    def _decode(self, final=False):
        torch = self.torch
        while True:
            step = 0 if self.steps == 0 else 1
            width = min(self.chunk[step], self.features.shape[-1] - self.cursor)
            if width < self.chunk[step] and not final:
                return
            if width < self.minimum[step]:
                return
            last = final and self.cursor + width >= self.features.shape[-1]
            chunk = self.features[:, :, max(0, self.cursor - self.pre_encode[step]):
                                  self.cursor + width]
            length = torch.tensor([chunk.shape[-1]], device=self.model.device)
            with torch.inference_mode():
                (self.prediction, texts, self.cache_channel, self.cache_time,
                 self.cache_len, self.hypotheses) = self.model.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=length,
                    cache_last_channel=self.cache_channel,
                    cache_last_time=self.cache_time,
                    cache_last_channel_len=self.cache_len,
                    keep_all_outputs=last,
                    previous_hypotheses=self.hypotheses,
                    previous_pred_out=self.prediction,
                    drop_extra_pre_encoded=None, return_transcription=True)
            self.cursor += self.shift[step]
            self.steps += 1
            value = texts[0]
            self._read(value.text if hasattr(value, "text") else str(value))
            if last:
                return

    def _read(self, value):
        tags = self.LANGUAGE_TAG.findall(value)
        if tags:
            self.detected = tags[-1][1:-1]
        self.text = " ".join(self.LANGUAGE_TAG.sub(" ", value).split())

    def audio(self, pcm):
        self.pending = np.concatenate((self.pending, to_float32(pcm)))
        self._extend_features()
        self._decode()
        return {"text": self.text, "language": self.detected}

    def finish(self):
        remainder = len(self.pending) % self.hop
        if remainder:
            self.pending = np.concatenate(
                (self.pending, np.zeros(self.hop - remainder, dtype=np.float32)))
        self._extend_features()
        # Pad out to a whole chunk so the last step can flush its right context.
        step = 0 if self.steps == 0 else 1
        missing = self.cursor + self.chunk[step] - self.features.shape[-1]
        if missing > 0:
            self.pending = np.zeros(missing * self.hop, dtype=np.float32)
            self._extend_features()
        self._decode(final=True)
        return {"text": self.text, "language": self.detected}


class VoskWorker:
    """Kaldi decoding graph; runs on CPU and streams at the native sample rate."""

    def __init__(self, checkpoint: str):
        import vosk

        # Kaldi logs through the C-level stderr, which Python redirection misses.
        vosk.SetLogLevel(-1)
        self.vosk = vosk
        self.model = vosk.Model(model_name=checkpoint)
        self.recognizer = None

    def metadata(self):
        from importlib.metadata import version
        return {"backend": "Vosk/Kaldi", "vosk": version("vosk")}

    def start(self, code, settings):
        self.recognizer = self.vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        self.code = code
        self.utterances = []

    def _result(self, partial):
        return {"text": " ".join([*self.utterances, partial]).strip(),
                "language": self.code}

    def _close_utterance(self, payload):
        text = json.loads(payload)["text"]
        if text:
            self.utterances.append(text)

    def audio(self, pcm):
        if self.recognizer is None:
            raise RuntimeError("No active session")
        # A true return means silence ended the utterance: Result() is then final
        # and the partial buffer restarts, so completed text has to be kept here.
        if self.recognizer.AcceptWaveform(pcm):
            self._close_utterance(self.recognizer.Result())
            return self._result("")
        return self._result(json.loads(self.recognizer.PartialResult())["partial"])

    def finish(self):
        if self.recognizer is None:
            raise RuntimeError("No active session")
        self._close_utterance(self.recognizer.FinalResult())
        result = self._result("")
        self.recognizer = None
        return result


WORKERS = {"qwen": QwenWorker, "nemotron": NemotronWorker, "vosk": VoskWorker}


def main():
    worker = None
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "load":
                    spec = request["model"]
                    worker = WORKERS[spec["family"]](spec["checkpoint"])
                    result = worker.metadata()
                elif worker is None:
                    raise RuntimeError("Load a model first")
                elif request["op"] == "start":
                    worker.start(request["language"], request["settings"])
                    result = {"ok": True}
                elif request["op"] == "audio":
                    result = worker.audio(base64.b64decode(request["pcm"], validate=True))
                elif request["op"] == "finish":
                    result = worker.finish()
                else:
                    raise ValueError(f"Unknown operation: {request['op']}")
            result["processing_ms"] = (time.perf_counter() - started) * 1000
            reply(result)
        except Exception as exc:  # noqa: BLE001 - return all worker failures to API
            traceback.print_exc(file=sys.stderr)
            reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
