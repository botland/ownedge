"""Unit tests for vLLM image pull progress helpers."""

import models


def test_docker_pull_progress_bytes_partial():
    current, total = models._docker_pull_progress_bytes(
        {"progressDetail": {"current": 500, "total": 1000}}
    )
    assert current == 500
    assert total == 1000


def test_docker_pull_overall_percent():
    layer_totals = {"a": 1000, "b": 1000}
    layer_current = {"a": 500, "b": 1000}
    pct = models._docker_pull_overall_percent(layer_totals, layer_current)
    assert pct == 75.0


def test_docker_pull_bytes_summary():
    summary = models._docker_pull_bytes_summary({"a": 1024}, {"a": 512})
    assert summary is not None
    assert "/" in summary


def test_write_and_read_vllm_pull_progress(env_defaults, monkeypatch):
    progress_path = f"{env_defaults['cache_dir']}/.inferedge-vllm-pull-progress"
    monkeypatch.setattr(models, "_VLLM_PULL_PROGRESS_FILE", progress_path)
    models._write_vllm_pull_progress(42.5, "1.0 GB/2.0 GB")
    progress = models.get_vllm_pull_progress()
    assert progress["percent"] == 42.5
    assert "GB" in progress["human"]
    models._clear_vllm_pull_progress()
    assert models.get_vllm_pull_progress() == {}


def test_format_vllm_pull_log():
    msg = models._format_vllm_pull_log(
        "Downloading",
        "layer-abc",
        "10MB/20MB",
        50.0,
        "10.0 MB/20.0 MB",
    )
    assert "vLLM pull" in msg
    assert "50.0%" in msg


def test_should_log_vllm_pull_progress_on_first_event():
    assert models._should_log_vllm_pull_progress(0, -1, None, None) is True


def test_configure_hf_download_env_sets_xet_disabled(tmp_path, monkeypatch):
    import os

    target = str(tmp_path / "model-cache")
    models._configure_hf_download_env(target)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HOME"].startswith(target)