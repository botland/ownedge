from unittest.mock import patch

import pytest

import gpu
from gpu import GpuDevice, GpuInfo


def test_is_quantized_detects_awq_gptq():
    assert gpu._is_quantized("casperhansen/llama-3-8b-instruct-awq")
    assert gpu._is_quantized("model-gptq-4bit")
    assert not gpu._is_quantized("meta-llama/Llama-3.1-8B-Instruct")


def test_param_billions_parses_model_ids():
    assert gpu._param_billions("meta-llama/Llama-3.1-8B-Instruct") == 8.0
    assert gpu._param_billions("mistral-7b") == 7.0
    assert gpu._param_billions("unknown-model") is None


def test_estimate_weight_gb_quantized_vs_fp16():
    fp16 = gpu._estimate_weight_gb("meta-llama/Llama-3.1-8B-Instruct")
    awq = gpu._estimate_weight_gb("casperhansen/llama-3-8b-instruct-awq")
    assert fp16 == 16.0
    assert awq == pytest.approx(5.36, rel=0.01)


def test_kv_cache_scales_with_context():
    small = gpu._kv_cache_gb("llama-8b", 2048)
    large = gpu._kv_cache_gb("llama-8b", 8192)
    assert large > small


def test_is_gpu_available_false_when_no_devices(gpu_unavailable):
    with patch.object(gpu, "get_gpu_info", return_value=gpu_unavailable):
        assert gpu.is_gpu_available() is False
        assert gpu.get_gpu_uuids() == []
        assert gpu.total_vram_mb() == 0


def test_check_vram_returns_none_without_gpu(gpu_unavailable):
    with patch.object(gpu, "get_gpu_info", return_value=gpu_unavailable):
        assert gpu.check_vram_for_model("meta-llama/Llama-3.1-8B-Instruct", 8192) is None


def test_check_vram_returns_message_when_too_small():
    tiny_gpu = GpuInfo(
        available=True,
        devices=[
            GpuDevice(
                index=0,
                uuid="GPU-tiny",
                name="Tiny GPU",
                total_vram_mb=4 * 1024,
                free_vram_mb=3 * 1024,
            )
        ],
    )
    with patch.object(gpu, "get_gpu_info", return_value=tiny_gpu):
        msg = gpu.check_vram_for_model("meta-llama/Llama-3.1-8B-Instruct", 8192, 0.85)
        assert msg is not None
        assert "VRAM likely insufficient" in msg


def test_cap_context_length_fits_large_gpu(gpu_with_24gb):
    with patch.object(gpu, "get_gpu_info", return_value=gpu_with_24gb):
        capped = gpu.cap_context_length("casperhansen/llama-3-8b-instruct-awq", 8192, 0.85)
        assert capped == 8192


def test_cap_context_length_reduces_for_tiny_gpu():
    tiny_gpu = GpuInfo(
        available=True,
        devices=[
            GpuDevice(
                index=0,
                uuid="GPU-tiny",
                name="Tiny GPU",
                total_vram_mb=4 * 1024,
                free_vram_mb=3 * 1024,
            )
        ],
    )
    with patch.object(gpu, "get_gpu_info", return_value=tiny_gpu):
        capped = gpu.cap_context_length("meta-llama/Llama-3.1-8B-Instruct", 8192, 0.85)
        assert capped <= 8192
        assert capped in gpu._CONTEXT_STEPS


def test_tune_gpu_settings_prefers_higher_util_when_needed(gpu_with_24gb):
    with patch.object(gpu, "get_gpu_info", return_value=gpu_with_24gb):
        ctx, util = gpu.tune_gpu_settings("casperhansen/llama-3-8b-instruct-awq", 8192, 0.85)
        assert ctx >= 512
        assert util >= 0.85