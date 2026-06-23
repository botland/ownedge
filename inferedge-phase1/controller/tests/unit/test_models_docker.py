from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException

import models
from tests.conftest import FakeContainer


def test_find_managed_vllm_containers(mock_docker_client, fake_vllm_container):
    containers = models.find_managed_vllm_containers()
    assert len(containers) == 1
    assert containers[0].id == fake_vllm_container.id
    mock_docker_client.containers.list.assert_called()


def test_is_vllm_container_running(mock_docker_client, fake_vllm_container):
    assert models.is_vllm_container_running() is True
    fake_vllm_container.status = "exited"
    assert models.is_vllm_container_running() is False


def test_get_vllm_load_hint(mock_docker_client, fake_vllm_container):
    fake_vllm_container._logs = b"INFO Loading weights...\nINFO Ready to serve\n"
    hint = models.get_vllm_load_hint(fake_vllm_container.id)
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
            "inferedge.config_hash": models.compute_config_hash(sample_desired),
            "inferedge.generation": "2",
        },
    )
    mock_docker_client.containers.list.return_value = [stale, running]

    stopped = await models.stop_vllm_if_needed(
        except_hash=models.compute_config_hash(sample_desired),
        except_generation=2,
    )
    assert stopped == 1
    assert stale.status == "removed"


def test_prune_exited_vllm_containers(mock_docker_client):
    exited = FakeContainer(status="exited", labels={"inferedge.managed": "true", "inferedge.component": "vllm"})
    running = FakeContainer(status="running", labels={"inferedge.managed": "true", "inferedge.component": "vllm"})
    mock_docker_client.containers.list.return_value = [exited, running]

    actions = models.prune_exited_vllm_containers()
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

    monkeypatch.setattr(models.httpx, "Client", FakeClient)
    actual = models._probe_vllm(sample_desired.model)
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
            raise models.httpx.RequestError("connection refused", request=MagicMock())

    monkeypatch.setattr(models.httpx, "Client", FakeClient)
    actual = models._probe_vllm(sample_desired.model)
    assert actual.health == "UNREACHABLE"


def test_get_deployment_status_sync_running(mock_docker_client, fake_vllm_container, sample_desired):
    with patch.object(models, "_probe_vllm") as probe:
        from schemas import ActualState

        probe.return_value = ActualState(
            health="HEALTHY",
            model_loaded=True,
            current_model=sample_desired.model,
        )
        actual, exits = models._get_deployment_status_sync(sample_desired.model, {})
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
    with patch.object(models, "find_managed_vllm_containers", return_value=[]):
        actions = models.heal_deployment_environment()
    assert any("orphan" in a for a in actions)
    assert orphan.status == "removed"