import pytest

from serving.load_errors import format_vllm_load_error, has_vllm_load_failure
from serving.types import apply_gpu_profile, compute_config_hash, normalize_model_key


def test_normalize_model_key():
    assert normalize_model_key("org/model-name") == "org--model-name"


def test_compute_config_hash_stable(sample_desired):
    h1 = compute_config_hash(sample_desired)
    h2 = compute_config_hash(sample_desired)
    assert h1 == h2
    assert len(h1) == 64

    changed = sample_desired.model_copy(update={"context_length": 4096})
    assert compute_config_hash(changed) != h1


def test_format_vllm_load_error_kv_cache_hint():
    record = {
        "exit_code": 1,
        "log_snippet": (
            "ValueError: The model's max seq len (2048) is larger than the "
            "maximum number of tokens that can be stored in KV cache (304)."
        ),
    }
    msg = format_vllm_load_error(record)
    assert "vLLM failed to load model" in msg
    assert "context_length" in msg


def test_format_vllm_load_error_exit_code_fallback():
    msg = format_vllm_load_error({"exit_code": 137, "log_snippet": ""})
    assert "code=137" in msg


def test_has_vllm_load_failure():
    assert has_vllm_load_failure("abc", {"config_hash": "abc", "exit_code": 1})
    assert not has_vllm_load_failure("abc", {"config_hash": "abc", "exit_code": 0})
    assert not has_vllm_load_failure("abc", {"config_hash": "other", "exit_code": 1})


def test_apply_gpu_profile_noop_when_disabled(sample_desired, monkeypatch):
    monkeypatch.setenv("GPU_PROFILE", "manual")
    result = apply_gpu_profile(sample_desired, sample_desired.model)
    assert result == sample_desired