"""JSON-lines model worker, run in a family-specific Python environment."""
from __future__ import annotations

import base64
import contextlib
import copy
import json
import sys
import time
import traceback
from typing import Any

import numpy as np


def reply(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


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
        self.model.streaming_transcribe(pcm, self.state)
        return {"text": self.state.text or "", "language": self.state.language}

    def finish(self):
        if self.state is None:
            raise RuntimeError("No active session")
        self.model.finish_streaming_transcribe(self.state)
        result = {"text": self.state.text or "", "language": self.state.language}
        self.state = None
        return result


class NemotronWorker:
    RIGHT_CONTEXT = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}

    def __init__(self, checkpoint: str):
        import torch
        import nemo.collections.asr as nemo_asr

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
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import EncDecCTCModelBPE

        cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        return EncDecCTCModelBPE.from_config_dict(cfg.preprocessor).to(self.model.device)

    def start(self, code, settings):
        chunk_ms = int(settings["chunk_ms"])
        left = self.model.encoder.att_context_size[0]
        self.model.encoder.set_default_att_context_size(
            [left, self.RIGHT_CONTEXT[chunk_ms]])
        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(language(code, "nemotron"))
        self.preprocessor = self._make_preprocessor()
        self.chunk_samples = 16 * chunk_ms
        self.pending = np.empty(0, dtype=np.float32)
        self.cache_channel, self.cache_time, self.cache_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1))
        self.hypotheses = self.prediction = None
        size = self.model.encoder.streaming_cfg.pre_encode_cache_size[1]
        features = self.model.cfg.preprocessor.features
        self.pre_cache = self.torch.zeros(
            (1, features, size), device=self.model.device)
        self.text = ""
        self.detected = None

    def _step(self, samples, final=False):
        torch = self.torch
        signal = torch.from_numpy(samples).unsqueeze(0).to(self.model.device)
        length = torch.tensor([len(samples)], device=self.model.device)
        processed, processed_len = self.preprocessor(
            input_signal=signal, length=length)
        processed = torch.cat([self.pre_cache, processed], dim=-1)
        processed_len += self.pre_cache.shape[-1]
        self.pre_cache = processed[:, :, -self.pre_cache.shape[-1]:]
        with torch.inference_mode():
            (self.prediction, texts, self.cache_channel, self.cache_time,
             self.cache_len, self.hypotheses) = self.model.conformer_stream_step(
                processed_signal=processed, processed_signal_length=processed_len,
                cache_last_channel=self.cache_channel,
                cache_last_time=self.cache_time,
                cache_last_channel_len=self.cache_len,
                keep_all_outputs=final,
                previous_hypotheses=self.hypotheses,
                previous_pred_out=self.prediction,
                drop_extra_pre_encoded=None, return_transcription=True)
        value = texts[0]
        self.text = value.text if hasattr(value, "text") else str(value)
        if self.text.startswith("<") and ">" in self.text:
            tag, self.text = self.text.split(">", 1)
            self.detected = tag[1:]
            self.text = self.text.lstrip()

    def audio(self, pcm):
        self.pending = np.concatenate((self.pending, pcm))
        # Keep the newest complete chunk buffered so finish() can mark the real
        # last chunk with keep_all_outputs=True (required to flush right context).
        while len(self.pending) > self.chunk_samples:
            chunk = self.pending[:self.chunk_samples]
            self.pending = self.pending[self.chunk_samples:]
            self._step(chunk)
        return {"text": self.text, "language": self.detected}

    def finish(self):
        if len(self.pending):
            chunk = np.pad(self.pending, (0, self.chunk_samples - len(self.pending)))
            self._step(chunk, final=True)
        self.pending = np.empty(0, dtype=np.float32)
        return {"text": self.text, "language": self.detected}


def main():
    worker = None
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "load":
                    spec = request["model"]
                    worker = (QwenWorker(spec["checkpoint"]) if spec["family"] == "qwen"
                              else NemotronWorker(spec["checkpoint"]))
                    result = worker.metadata()
                elif worker is None:
                    raise RuntimeError("Load a model first")
                elif request["op"] == "start":
                    worker.start(request["language"], request["settings"])
                    result = {"ok": True}
                elif request["op"] == "audio":
                    raw_pcm = base64.b64decode(request["pcm"], validate=True)
                    pcm = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / 32768.0
                    result = worker.audio(pcm)
                elif request["op"] == "finish":
                    result = worker.finish()
                else:
                    raise ValueError(f"Unknown operation: {request['op']}")
            result["processing_ms"] = (time.perf_counter() - started) * 1000
            reply(result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
