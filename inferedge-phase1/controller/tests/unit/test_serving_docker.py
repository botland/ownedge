"""Unit tests for Docker vLLM serving backend."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from docker.errors import DockerException

import serving.docker_vllm as docker_vllm
from exceptions import DockerError, TransientDockerError
from serving.types import compute_config_hash
from tests.conftest import FakeContainer


def test_build_labels(sample_desired):
    labels = docker_vllm._build_labels("org--model", "hash123", 7, ["GPU-a", "GPU-b"])
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
    assert docker_vllm._detect_quantization(str(model_dir), "any/model") == "awq"


def test_detect_quantization_from_model_id():
    assert docker_vllm._detect_quantization("/missing", "vendor/llama-awq-4bit") == "awq"
    assert docker_vllm._detect_quantization("/missing", "plain/model") is None


def test_format_bytes():
    assert docker_vllm._format_bytes(512) == "512 B"
    assert "KB" in docker_vllm._format_bytes(2048)
    assert "MB" in docker_vllm._format_bytes(2 * 1024**2)
    assert "GB" in docker_vllm._format_bytes(3 * 1024**3)


def test_model_probe_match_variants():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    assert docker_vllm._model_probe_match(model_id, [model_id])
    assert docker_vllm._model_probe_match(model_id, ["meta-llama--Llama-3.1-8B-Instruct"])
    assert docker_vllm._model_probe_match(model_id, ["Llama-3.1-8B-Instruct"])
    assert not docker_vllm._model_probe_match(model_id, ["other/model"])


def test_docker_error_is_transient():
    assert docker_vllm._docker_error_is_transient(DockerException("409 Conflict"))
    assert docker_vllm._docker_error_is_transient(DockerException("name already in use"))
    assert not docker_vllm._docker_error_is_transient(DockerException("connection refused"))


def test_raise_docker_error_transient():
    with pytest.raises(TransientDockerError):
        docker_vllm._raise_docker_error("start", DockerException("409 conflict"))


def test_raise_docker_error_permanent():
    with pytest.raises(DockerError):
        docker_vllm._raise_docker_error("start", DockerException("permission denied"))


def test_vllm_cli_style_flag_when_entrypoint_unknown(monkeypatch):
    client = MagicMock()
    client.images.get.side_effect = DockerException("missing")
    monkeypatch.setattr(docker_vllm, "_vllm_cli_style_cache", None)
    assert docker_vllm._vllm_cli_style(client) == "flag"


def test_build_vllm_command_flag_style(monkeypatch, sample_desired, tmp_path):
    client = MagicMock()
    monkeypatch.setattr(docker_vllm, "_vllm_cli_style_cache", ("img", "flag"))
    cmd = docker_vllm._build_vllm_command(client, str(tmp_path), sample_desired.model, sample_desired)
    assert cmd[0] == "--model"
    assert "--max-model-len" in cmd
    assert str(sample_desired.context_length) in cmd


def test_exit_record_changed_detects_diff():
    stored = {"container_id": "a", "exit_code": 1, "generation": 1, "config_hash": "h", "log_snippet": "x"}
    same = dict(stored)
    different = dict(stored, exit_code=2)
    assert not docker_vllm._exit_record_changed(stored, same)
    assert docker_vllm._exit_record_changed(stored, different)


def test_docker_pull_progress_bytes_partial():
    current, total = docker_vllm._docker_pull_progress_bytes(
        {"progressDetail": {"current": 500, "total": 1000}}
    )
    assert current == 500
    assert total == 1000


def test_docker_pull_overall_percent():
    layer_totals = {"a": 1000, "b": 1000}
    layer_current = {"a": 500, "b": 1000}
    pct = docker_vllm._docker_pull_overall_percent(layer_totals, layer_current)
    assert pct == 75.0


def test_docker_pull_bytes_summary():
    summary = docker_vllm._docker_pull_bytes_summary({"a": 1024}, {"a": 512})
    assert summary is not None
    assert "/" in summary


def test_write_and_read_vllm_pull_progress(env_defaults, monkeypatch):
    progress_path = f"{env_defaults['cache_dir']}/.inferedge-vllm-pull-progress"
    monkeypatch.setattr(docker_vllm, "_VLLM_PULL_PROGRESS_FILE", progress_path)
    docker_vllm._write_vllm_pull_progress(42.5, "1.0 GB/2.0 GB")
    progress = docker_vllm.get_vllm_pull_progress()
    assert progress["percent"] == 42.5
    assert "GB" in progress["human"]
    docker_vllm._clear_vllm_pull_progress()
    assert docker_vllm.get_vllm_pull_progress() == {}


def test_format_vllm_pull_log():
    msg = docker_vllm._format_vllm_pull_log(
        "Downloading",
        "layer-abc",
        "10MB/20MB",
        50.0,
        "10.0 MB/20.0 MB",
    )
    assert "vLLM pull" in msg
    assert "50.0%" in msg


def test_should_log_vllm_pull_progress_on_first_event():
    assert docker_vllm._should_log_vllm_pull_progress(0, -1, None, None) is True


def test_find_managed_vllm_containers(mock_docker_client, fake_vllm_container):
    containers = docker_vllm.find_managed_vllm_containers()
    assert len(containers) == 1
    assert containers[0].id == fake_vllm_container.id
    mock_docker_client.containers.list.assert_called()


def test_is_vllm_container_running(mock_docker_client, fake_vllm_container):
    assert docker_vllm.is_vllm_container_running() is True
    fake_vllm_container.status = "exited"
    assert docker_vllm.is_vllm_container_running() is False


def test_get_vllm_load_hint(mock_docker_client, fake_vllm_container):
    fake_vllm_container._logs = b"INFO Loading weights...\nINFO Ready to serve\n"
    hint = docker_vllm.get_vllm_load_hint(fake_vllm_container.id)
    assert hint is not None
    assert "Ready" in hint or "Loading" in hint


@pytest.mark.asyncio
async def test_stop_vllm_if_needed_removes_stale(fresh_state, mock_docker_client, sample_desired):
    stale = FakeContainer(
        id="stale-id",
        name="inferedge-vllm-gen1",
        status="exited",
        labels={
            "inferedge.managed": "true",
            "inferedge.component": "vllm",
            "inferedge.config_hash": "old-hash",
            "inferedge.generation": "1",
        },
        attrs={"State": {"ExitCode": 1}},
    )
    running = FakeContainer(
        id="running-id",
        name="inferedge-vllm-gen2",
        status="running",
        labels={
            "inferedge.managed": "true",
            "inferedge.component": "vllm",
            "inferedge.config_hash": compute_config_hash(sample_desired),
            "inferedge.generation": "2",
        },
    )
    mock_docker_client.containers.list.return_value = [stale, running]

    stopped = await docker_vllm.stop_vllm_if_needed(
        except_hash=compute_config_hash(sample_desired),
        except_generation=2,
    )
    assert stopped == 1
    assert stale.status == "removed"


def test_prune_exited_vllm_containers(mock_docker_client):
    exited = FakeContainer(status="exited", labels={"inferedge.managed": "true", "inferedge.component": "vllm"})
    running = FakeContainer(status="running", labels={"inferedge.managed": "true", "inferedge.component": "vllm"})
    mock_docker_client.containers.list.return_value = [exited, running]

    actions = docker_vllm.prune_exited_vllm_containers()
    assert any("removed" in a for a in actions)
    assert exited.status == "removed"


def test_probe_vllm_healthy(monkeypatch, sample_desired, mock_docker_client):
    class FakeResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            if url.endswith("/health"):
                return FakeResponse(200)
            return FakeResponse(200, {"data": [{"id": sample_desired.model}]})

    monkeypatch.setattr(docker_vllm.httpx, "Client", FakeClient)
    actual = docker_vllm._probe_vllm(sample_desired.model)
    assert actual.health == "HEALTHY"
    assert actual.model_loaded is True


def test_probe_vllm_unreachable_without_container(monkeypatch, sample_desired, mock_docker_client):
    mock_docker_client.containers.list.return_value = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            raise httpx.RequestError("connection refused", request=MagicMock())

    monkeypatch.setattr(docker_vllm.httpx, "Client", FakeClient)
    actual = docker_vllm._probe_vllm(sample_desired.model)
    assert actual.health == "UNREACHABLE"


def test_get_deployment_status_sync_running(mock_docker_client, fake_vllm_container, sample_desired):
    with patch.object(docker_vllm, "_probe_vllm") as probe:
        from schemas import ActualState

        probe.return_value = ActualState(
            health="HEALTHY",
            model_loaded=True,
            current_model=sample_desired.model,
        )
        actual, exits = docker_vllm._get_deployment_status_sync(sample_desired.model, {})
        assert actual.health == "HEALTHY"
        assert actual.container_id == fake_vllm_container.id
        assert exits == []


def test_heal_deployment_environment_removes_orphans(mock_docker_client):
    orphan = FakeContainer(
        name="inferedge-vllm-gen9",
        status="created",
        labels={},
    )
    mock_docker_client.containers.list.return_value = [orphan]
    with patch.object(docker_vllm, "find_managed_vllm_containers", return_value=[]):
        actions = docker_vllm.heal_deployment_environment()
    assert any("orphan" in a for a in actions)
    assert orphan.status == "removed"