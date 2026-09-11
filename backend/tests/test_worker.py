import pytest

from stt_lab.worker import GIB, qwen_gpu_memory_utilization


def test_qwen_memory_budget_uses_currently_free_memory(monkeypatch):
    monkeypatch.delenv("STT_QWEN_GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.delenv("STT_QWEN_GPU_HEADROOM_GIB", raising=False)

    assert qwen_gpu_memory_utilization(16 * GIB, 24 * GIB) == pytest.approx(15 / 24)


def test_qwen_memory_budget_keeps_configured_cap(monkeypatch):
    monkeypatch.setenv("STT_QWEN_GPU_MEMORY_UTILIZATION", "0.7")
    monkeypatch.setenv("STT_QWEN_GPU_HEADROOM_GIB", "0.5")

    assert qwen_gpu_memory_utilization(23 * GIB, 24 * GIB) == 0.7


@pytest.mark.parametrize(
    ("name", "value"),
    [("STT_QWEN_GPU_MEMORY_UTILIZATION", "0"),
     ("STT_QWEN_GPU_MEMORY_UTILIZATION", "invalid"),
     ("STT_QWEN_GPU_HEADROOM_GIB", "-1")],
)
def test_qwen_memory_budget_rejects_invalid_configuration(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError):
        qwen_gpu_memory_utilization(16 * GIB, 24 * GIB)
