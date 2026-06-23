import json
from unittest.mock import MagicMock

import pytest
from docker.errors import DockerException

import models
from schemas import DesiredState


def test_normalize_model_key():
    assert models.normalize_model_key("org/model-name") == "org--model-name"


def test_compute_config_hash_stable(sample_desired):
    h1 = models.compute_config_hash(sample_desired)
    h2 = models.compute_config_hash(sample_desired)
    assert h1 == h2
    assert len(h1) == 64

    changed = sample_desired.model_copy(update={"context_length": 4096})
    assert models.compute_config_hash(changed) != h1


def test_build_labels(sample_desired):
    labels = models._build_labels("org--model", "hash123", 7, ["GPU-a", "GPU-b"])
    assert labels["inferedge.managed"] == "true"
    assert labels["inferedge.component"] == "vllm"
    assert labels["inferedge.config_hash"] == "hash123"
    assert labels["inferedge.generation"] == "7"
    assert labels["inferedge.gpu_ids"] == "GPU-a,GPU-b"


def test_detect_quantization_from_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "awq"}})
    )
    assert models._detect_quantization(str(model_dir), "any/model") == "awq"


def test_detect_quantization_from_model_id():
    assert models._detect_quantization("/missing", "vendor/llama-awq-4bit") == "awq"
    assert models._detect_quantization("/missing", "plain/model") is None


def test_format_bytes():
    assert models._format_bytes(512) == "512 B"
    assert "KB" in models._format_bytes(2048)
    assert "MB" in models._format_bytes(2 * 1024**2)
    assert "GB" in models._format_bytes(3 * 1024**3)


def test_is_likely_gated():
    assert models._is_likely_gated("meta-llama/Llama-3.1-8B-Instruct")
    assert models._is_likely_gated("google/gemma-7b")
    assert not models._is_likely_gated("casperhansen/llama-3-8b-instruct-awq")


def test_should_download_file_filters_gguf_and_original():
    assert models._should_download_file("model.safetensors")
    assert models._should_download_file("tokenizer.json")
    assert not models._should_download_file("weights.gguf")
    assert not models._should_download_file("original/model.safetensors")


def test_format_vllm_load_error_kv_cache_hint():
    record = {
        "exit_code": 1,
        "log_snippet": (
            "ValueError: The model's max seq len (2048) is larger than the "
            "maximum number of tokens that can be stored in KV cache (304)."
        ),
    }
    msg = models.format_vllm_load_error(record)
    assert "vLLM failed to load model" in msg
    assert "context_length" in msg


def test_format_vllm_load_error_exit_code_fallback():
    msg = models.format_vllm_load_error({"exit_code": 137, "log_snippet": ""})
    assert "code=137" in msg


def test_has_vllm_load_failure():
    assert models.has_vllm_load_failure("abc", {"config_hash": "abc", "exit_code": 1})
    assert not models.has_vllm_load_failure("abc", {"config_hash": "abc", "exit_code": 0})
    assert not models.has_vllm_load_failure("abc", {"config_hash": "other", "exit_code": 1})


def test_model_probe_match_variants():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    assert models._model_probe_match(model_id, [model_id])
    assert models._model_probe_match(model_id, ["meta-llama--Llama-3.1-8B-Instruct"])
    assert models._model_probe_match(model_id, ["Llama-3.1-8B-Instruct"])
    assert not models._model_probe_match(model_id, ["other/model"])


def test_docker_error_is_transient():
    assert models._docker_error_is_transient(DockerException("409 Conflict"))
    assert models._docker_error_is_transient(DockerException("name already in use"))
    assert not models._docker_error_is_transient(DockerException("connection refused"))


def test_raise_docker_error_transient():
    with pytest.raises(models.TransientDockerError):
        models._raise_docker_error("start", DockerException("409 conflict"))


def test_raise_docker_error_permanent():
    with pytest.raises(models.DockerError):
        models._raise_docker_error("start", DockerException("permission denied"))


def test_download_sizes_grew():
    before = {"/tmp/a.part": 1000}
    after = {"/tmp/a.part": 200000}
    assert models._download_sizes_grew(before, after)
    assert not models._download_sizes_grew(after, after)


def test_apply_gpu_profile_noop_when_disabled(sample_desired, monkeypatch):
    monkeypatch.setenv("GPU_PROFILE", "manual")
    result = models.apply_gpu_profile(sample_desired, sample_desired.model)
    assert result == sample_desired


def test_vllm_cli_style_flag_when_entrypoint_unknown(monkeypatch):
    client = MagicMock()
    client.images.get.side_effect = DockerException("missing")
    monkeypatch.setattr(models, "_vllm_cli_style_cache", None)
    assert models._vllm_cli_style(client) == "flag"


def test_build_vllm_command_flag_style(monkeypatch, sample_desired, tmp_path):
    client = MagicMock()
    monkeypatch.setattr(models, "_vllm_cli_style_cache", ("img", "flag"))
    cmd = models._build_vllm_command(client, str(tmp_path), sample_desired.model, sample_desired)
    assert cmd[0] == "--model"
    assert "--max-model-len" in cmd
    assert str(sample_desired.context_length) in cmd


def test_exit_record_changed_detects_diff():
    stored = {"container_id": "a", "exit_code": 1, "generation": 1, "config_hash": "h", "log_snippet": "x"}
    same = dict(stored)
    different = dict(stored, exit_code=2)
    assert not models._exit_record_changed(stored, same)
    assert models._exit_record_changed(stored, different)