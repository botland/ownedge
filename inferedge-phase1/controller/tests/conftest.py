"""Shared fixtures for InferEdge controller tests."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Must be set before controller modules that read env at import are loaded in tests.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_inferedge.db")


@pytest.fixture
def tmp_cache_dir(tmp_path):
    cache = tmp_path / "models_cache"
    cache.mkdir()
    return str(cache)


@pytest.fixture
def env_defaults(monkeypatch, tmp_db_path, tmp_cache_dir):
    """Isolate controller env for each test."""
    monkeypatch.setenv("SQLITE_DB_PATH", tmp_db_path)
    monkeypatch.setenv("LOCAL_MODEL_CACHE", tmp_cache_dir)
    monkeypatch.setenv("MODEL_CACHE_HOST", tmp_cache_dir)
    monkeypatch.setenv("APPLIANCE_ID", "test-appliance-001")
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "inferedge-test")
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "test-token")
    monkeypatch.setenv("DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    monkeypatch.setenv("DEFAULT_CONTEXT", "8192")
    monkeypatch.setenv("GPU_UTILIZATION", "0.85")
    monkeypatch.setenv("RECONCILE_INTERVAL_SEC", "0.05")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.delenv("GPU_PROFILE", raising=False)
    return {
        "db_path": tmp_db_path,
        "cache_dir": tmp_cache_dir,
    }


@pytest_asyncio.fixture
async def fresh_state(env_defaults):
    """Reset SQLite module globals and return a migrated empty database."""
    import state

    await state.close_db()
    state._db = None
    state.DB_PATH = env_defaults["db_path"]
    await state.migrate()
    yield state
    await state.close_db()
    state._db = None


@pytest.fixture
def initialized_db(env_defaults):
    """Sync wrapper for tests that do not run under asyncio."""

    async def _setup():
        import state

        await state.close_db()
        state._db = None
        state.DB_PATH = env_defaults["db_path"]
        await state.migrate()

    async def _teardown():
        import state

        await state.close_db()
        state._db = None

    asyncio.run(_setup())
    yield
    asyncio.run(_teardown())


@pytest.fixture
def gpu_with_24gb():
    from gpu import GpuDevice, GpuInfo

    return GpuInfo(
        available=True,
        devices=[
            GpuDevice(
                index=0,
                uuid="GPU-test-uuid-0001",
                name="NVIDIA Test GPU",
                total_vram_mb=24 * 1024,
                free_vram_mb=20 * 1024,
            )
        ],
    )


@pytest.fixture
def gpu_unavailable():
    from gpu import GpuInfo

    return GpuInfo(available=False, devices=[], error="no gpu")


@pytest.fixture
def sample_desired():
    from schemas import DesiredState

    return DesiredState(
        model="meta-llama/Llama-3.1-8B-Instruct",
        context_length=8192,
        gpu_utilization=0.85,
    )


@dataclass
class FakeContainer:
    id: str = "container-deadbeef"
    name: str = "inferedge-vllm-gen1"
    status: str = "running"
    labels: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=lambda: {"State": {"ExitCode": 0}})
    _logs: bytes = b"INFO vLLM ready\n"

    def reload(self) -> None:
        return None

    def stop(self, timeout: int = 30) -> None:
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.status = "removed"

    def logs(self, tail: int = 200) -> bytes:
        return self._logs[-8000:]


@pytest.fixture
def fake_vllm_container(sample_desired):
    import models

    config_hash = models.compute_config_hash(sample_desired)
    return FakeContainer(
        id="abc123container",
        name="inferedge-vllm-gen3",
        status="running",
        labels={
            "inferedge.managed": "true",
            "inferedge.component": "vllm",
            "inferedge.appliance_id": "test-appliance-001",
            "inferedge.model_key": models.normalize_model_key(sample_desired.model),
            "inferedge.config_hash": config_hash,
            "inferedge.generation": "3",
            "inferedge.gpu_ids": "GPU-test-uuid-0001",
        },
    )


@pytest.fixture
def mock_docker_client(monkeypatch, fake_vllm_container):
    """Patch models._get_client with a lightweight fake Docker client."""
    import models

    client = MagicMock()
    client.containers.list.return_value = [fake_vllm_container]
    client.containers.get.return_value = fake_vllm_container
    client.images.get.side_effect = Exception("image not found")
    client.networks.get.return_value = MagicMock()

    monkeypatch.setattr(models, "_docker_client", None)
    monkeypatch.setattr(models, "_get_client", lambda: client)
    return client


@pytest.fixture
def patch_reconcile_externals(monkeypatch):
    """Disable GPU profiling and host GPU checks during reconciler tests."""
    import gpu
    import models

    monkeypatch.setattr(models, "apply_gpu_profile", lambda desired, _model_id: desired)
    monkeypatch.setattr(gpu, "is_gpu_available", lambda: True)
    monkeypatch.setattr(gpu, "check_vram_for_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(models, "get_vllm_load_hint", lambda *args, **kwargs: None)


@pytest.fixture
def healthy_actual(sample_desired, fake_vllm_container):
    from schemas import ActualState

    import models

    return ActualState(
        model_loaded=True,
        current_model=sample_desired.model,
        container_id=fake_vllm_container.id,
        health="HEALTHY",
        config_hash=models.compute_config_hash(sample_desired),
        generation=3,
        gpu_ids="GPU-test-uuid-0001",
    )