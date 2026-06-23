from unittest.mock import MagicMock, patch

import pytest

from serving import ServingStack, get_serving_stack
from serving.docker_vllm import DockerVllmServingBackend


def test_get_serving_stack_litellm_vllm_default(monkeypatch):
    monkeypatch.delenv("COMPUTE_BACKEND", raising=False)
    stack = get_serving_stack()
    assert stack.mode == "litellm_vllm"
    assert isinstance(stack.backend, DockerVllmServingBackend)
    assert stack.scheduler is None


def test_get_serving_stack_local_alias(monkeypatch):
    monkeypatch.setenv("COMPUTE_BACKEND", "local")
    stack = get_serving_stack()
    assert stack.mode == "litellm_vllm"


def test_get_serving_stack_unknown_backend(monkeypatch):
    monkeypatch.setenv("COMPUTE_BACKEND", "kubernetes")
    with pytest.raises(ValueError, match="Unknown COMPUTE_BACKEND"):
        get_serving_stack()


def test_ray_cluster_scheduler_lifecycle(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray://test-head:10001")
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True

    with patch.dict("sys.modules", {"ray": mock_ray}):
        from compute.ray_cluster import RayClusterScheduler

        sched = RayClusterScheduler()
        assert not sched.is_ready()
        sched.start()
        assert sched.is_ready()
        sched.shutdown()
        mock_ray.init.assert_called_once()
        mock_ray.shutdown.assert_called_once()