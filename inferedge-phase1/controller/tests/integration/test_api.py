import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import state
from schemas import ActualState, ApplianceState
from serving import ServingStack
from serving.docker_vllm import DockerVllmServingBackend


@pytest.fixture
def api_client(initialized_db, env_defaults, monkeypatch):
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "test-token")
    monkeypatch.setenv("APPLIANCE_ID", "test-appliance-001")

    mock_stack = ServingStack("litellm_vllm", MagicMock(spec=DockerVllmServingBackend), None)

    with (
        patch("main.get_serving_stack", return_value=mock_stack),
        patch("main.Reconciler") as reconciler_cls,
        patch("main.API_TOKEN", "test-token"),
        patch("main.APPLIANCE_ID", "test-appliance-001"),
    ):
        reconciler = MagicMock()
        reconciler.run_loop = AsyncMock()
        reconciler.stop = MagicMock()
        reconciler_cls.return_value = reconciler

        from main import app

        with TestClient(app) as client:
            yield client, reconciler


@pytest.mark.integration
def test_health_endpoint(api_client):
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["compute_backend"] == "litellm_vllm"
    assert body["serving_ready"] is True
    assert body["scheduler_ready"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_endpoint_immediate(fresh_state, api_client):
    client, _ = api_client
    await state.set_appliance_state(ApplianceState.BOOT, last_reconcile_ts=None)
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "BOOT"
    assert body["appliance_id"] == "test-appliance-001"
    assert "desired" in body
    assert "actual" in body


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_reflects_cached_actual(fresh_state, api_client):
    client, _ = api_client
    actual = ActualState(health="DOWNLOADING", download_bytes=999)
    await state.set_appliance_state(
        ApplianceState.RECONCILING,
        last_error="Downloading model",
        last_reconcile_ts=42.0,
        actual=actual,
    )
    resp = client.get("/status")
    body = resp.json()
    assert body["state"] == "RECONCILING"
    assert body["last_reconcile_ts"] == 42.0
    assert body["actual"]["health"] == "DOWNLOADING"
    assert body["actual"]["download_bytes"] == 999


@pytest.mark.integration
def test_models_load_requires_token(api_client):
    client, _ = api_client
    resp = client.post("/models/load", json={"model": "llama-3.1-8b"})
    assert resp.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_load_queues_intent(fresh_state, api_client):
    client, _ = api_client
    await state.seed_defaults()
    resp = client.post(
        "/models/load",
        json={"model": "casperhansen/llama-3-8b-instruct-awq", "context_length": 2048},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert "sequence_id" in body

    db = await state.get_db()
    async with db.execute("SELECT processed FROM intent_log WHERE sequence_id = ?", (body["sequence_id"],)) as cur:
        row = await cur.fetchone()
    assert row[0] == 0


@pytest.mark.integration
def test_models_load_without_token_when_disabled(initialized_db, env_defaults, monkeypatch):
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "")
    mock_stack = ServingStack("litellm_vllm", MagicMock(), None)

    with (
        patch("main.get_serving_stack", return_value=mock_stack),
        patch("main.Reconciler") as reconciler_cls,
        patch("main.API_TOKEN", ""),
    ):
        reconciler = MagicMock()
        reconciler.run_loop = AsyncMock()
        reconciler.stop = MagicMock()
        reconciler_cls.return_value = reconciler
        from main import app

        with TestClient(app) as client:
            resp = client.post("/models/load", json={"model": "llama-3.1-8b"})
            assert resp.status_code == 200