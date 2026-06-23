import json
import os

import pytest

import artifacts
from exceptions import ArtifactError
from serving.types import normalize_model_key


def _write_shard(target: str, name: str, size: int = 1024) -> None:
    path = os.path.join(target, name)
    os.makedirs(os.path.dirname(path) or target, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


def test_cache_has_weights_complete_shards(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model-00001-of-00002.safetensors")
    _write_shard(target, "model-00002-of-00002.safetensors")
    assert artifacts._cache_has_weights(target)


def test_cache_has_weights_incomplete_shards(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model-00001-of-00002.safetensors")
    assert not artifacts._cache_has_weights(target)


def test_cache_has_weights_single_file(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model.safetensors", size=2048)
    assert artifacts._cache_has_weights(target)


def test_cache_has_weights_ignores_part_files(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model-00001-of-00001.safetensors.part")
    assert not artifacts._cache_has_weights(target)


def test_json_file_valid_and_corrupt(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps({"model_type": "llama"}))
    assert artifacts._json_file_valid(str(good))

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not artifacts._json_file_valid(str(bad))


def test_file_looks_corrupt_oversized_tokenizer(tmp_path):
    assert artifacts._file_looks_corrupt("tokenizer.json", 600 * 1024 * 1024)
    assert not artifacts._file_looks_corrupt("model-00001-of-00002.safetensors", 5 * 1024**3)


def test_validate_model_cache_detects_missing_metadata(tmp_path):
    target = str(tmp_path / "org--model")
    os.makedirs(target)
    _write_shard(target, "config.json")
    missing = artifacts._validate_model_cache("org/model", target)
    assert "tokenizer_config.json" in missing
    assert "tokenizer.json" in missing


def test_get_cache_stats_counts_bytes(tmp_path, env_defaults):
    model_id = "org/model"
    target = os.path.join(env_defaults["cache_dir"], normalize_model_key(model_id))
    _write_shard(target, "model-00001-of-00001.safetensors", size=4096)
    with open(os.path.join(target, ".inferedge-download-progress"), "w", encoding="utf-8") as f:
        f.write("[1/1] model.bin")

    stats = artifacts.get_cache_stats(model_id, env_defaults["cache_dir"])
    assert stats["bytes"] == 4096
    assert stats["weight_files"] == 1
    assert stats["current_file"] == "[1/1] model.bin"


def test_heal_download_environment_removes_locks_and_xet(tmp_path):
    target = str(tmp_path / "cache")
    lock_dir = os.path.join(target, ".hf_home", "hub", "nested")
    os.makedirs(lock_dir)
    lock_file = os.path.join(lock_dir, "file.lock")
    open(lock_file, "w").close()
    xet_dir = os.path.join(target, ".hf_home", "xet")
    os.makedirs(xet_dir)
    open(os.path.join(xet_dir, "chunk"), "w").close()

    artifacts.heal_download_environment(target)
    assert not os.path.exists(lock_file)
    assert not os.path.exists(xet_dir)


def test_check_disk_space_raises_when_low(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MIN_FREE_DISK_GB", 100000.0)
    with pytest.raises(ArtifactError, match="Only"):
        artifacts._check_disk_space(str(tmp_path))


def test_ensure_artifact_uses_existing_cache(tmp_path, env_defaults, monkeypatch):
    model_id = "org/model"
    target = os.path.join(env_defaults["cache_dir"], normalize_model_key(model_id))
    _write_shard(target, "model-00001-of-00001.safetensors", size=2048)
    for name, payload in (
        ("config.json", {"model_type": "llama"}),
        ("tokenizer_config.json", {"model_max_length": 2048}),
        ("tokenizer.json", {"version": "1.0"}),
        ("special_tokens_map.json", {}),
    ):
        with open(os.path.join(target, name), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    path = artifacts.ensure_artifact(model_id, env_defaults["cache_dir"])
    assert path == target


def test_hf_auth_error_for_gated_without_token(monkeypatch):
    monkeypatch.setattr(artifacts, "HF_TOKEN", "")
    with pytest.raises(ArtifactError, match="HF auth required"):
        artifacts._verify_hf_access("meta-llama/Llama-3.1-8B-Instruct")


def test_is_likely_gated():
    assert artifacts._is_likely_gated("meta-llama/Llama-3.1-8B-Instruct")
    assert artifacts._is_likely_gated("google/gemma-7b")
    assert not artifacts._is_likely_gated("casperhansen/llama-3-8b-instruct-awq")


def test_should_download_file_filters_gguf_and_original():
    assert artifacts._should_download_file("model.safetensors")
    assert artifacts._should_download_file("tokenizer.json")
    assert not artifacts._should_download_file("weights.gguf")
    assert not artifacts._should_download_file("original/model.safetensors")


def test_download_sizes_grew():
    before = {"/tmp/a.part": 1000}
    after = {"/tmp/a.part": 200000}
    assert artifacts._download_sizes_grew(before, after)
    assert not artifacts._download_sizes_grew(after, after)


def test_configure_hf_download_env_sets_xet_disabled(tmp_path, monkeypatch):
    target = str(tmp_path / "model-cache")
    artifacts._configure_hf_download_env(target)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HOME"].startswith(target)