import json
import os

import pytest

import models
from exceptions import ArtifactError


def _write_shard(target: str, name: str, size: int = 1024) -> None:
    path = os.path.join(target, name)
    os.makedirs(os.path.dirname(path) or target, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


def test_cache_has_weights_complete_shards(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model-00001-of-00002.safetensors")
    _write_shard(target, "model-00002-of-00002.safetensors")
    assert models._cache_has_weights(target)


def test_cache_has_weights_incomplete_shards(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model-00001-of-00002.safetensors")
    assert not models._cache_has_weights(target)


def test_cache_has_weights_single_file(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model.safetensors", size=2048)
    assert models._cache_has_weights(target)


def test_cache_has_weights_ignores_part_files(tmp_path):
    target = str(tmp_path / "org--model")
    _write_shard(target, "model-00001-of-00001.safetensors.part")
    assert not models._cache_has_weights(target)


def test_json_file_valid_and_corrupt(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps({"model_type": "llama"}))
    assert models._json_file_valid(str(good))

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not models._json_file_valid(str(bad))


def test_file_looks_corrupt_oversized_tokenizer(tmp_path):
    assert models._file_looks_corrupt("tokenizer.json", 600 * 1024 * 1024)
    assert not models._file_looks_corrupt("model-00001-of-00002.safetensors", 5 * 1024**3)


def test_validate_model_cache_detects_missing_metadata(tmp_path):
    target = str(tmp_path / "org--model")
    os.makedirs(target)
    _write_shard(target, "config.json")
    missing = models._validate_model_cache("org/model", target)
    assert "tokenizer_config.json" in missing
    assert "tokenizer.json" in missing


def test_get_cache_stats_counts_bytes(tmp_path, env_defaults):
    model_id = "org/model"
    target = os.path.join(env_defaults["cache_dir"], models.normalize_model_key(model_id))
    _write_shard(target, "model-00001-of-00001.safetensors", size=4096)
    with open(os.path.join(target, ".inferedge-download-progress"), "w", encoding="utf-8") as f:
        f.write("[1/1] model.bin")

    stats = models.get_cache_stats(model_id, env_defaults["cache_dir"])
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

    models.heal_download_environment(target)
    assert not os.path.exists(lock_file)
    assert not os.path.exists(xet_dir)


def test_check_disk_space_raises_when_low(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MIN_FREE_DISK_GB", 100000.0)
    with pytest.raises(ArtifactError, match="Only"):
        models._check_disk_space(str(tmp_path))


def test_ensure_artifact_uses_existing_cache(tmp_path, env_defaults, monkeypatch):
    model_id = "org/model"
    target = os.path.join(env_defaults["cache_dir"], models.normalize_model_key(model_id))
    _write_shard(target, "model-00001-of-00001.safetensors", size=2048)
    for name, payload in (
        ("config.json", {"model_type": "llama"}),
        ("tokenizer_config.json", {"model_max_length": 2048}),
        ("tokenizer.json", {"version": "1.0"}),
        ("special_tokens_map.json", {}),
    ):
        with open(os.path.join(target, name), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    path = models.ensure_artifact(model_id, env_defaults["cache_dir"])
    assert path == target


def test_hf_auth_error_for_gated_without_token(monkeypatch):
    monkeypatch.setattr(models, "HF_TOKEN", "")
    with pytest.raises(ArtifactError, match="HF auth required"):
        models._verify_hf_access("meta-llama/Llama-3.1-8B-Instruct")