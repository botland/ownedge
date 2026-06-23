"""Integration tests for reconciler + SQLite with external systems mocked."""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

import artifacts
import state
from serving.types import compute_config_hash
from exceptions import TransientDockerError
from reconciler import Reconciler
from schemas import ActualState, ApplianceState, DesiredState


@pytest_asyncio.fixture
async def ready_reconciler(fresh_state, serving_backend):
    await fresh_state.seed_defaults()
    return Reconciler("integration-appliance", serving_backend=serving_backend)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_reconcile_cycle_persists_ready_state(
    ready_reconciler, sample_desired, patch_reconcile_externals
):
    config_hash = compute_config_hash(sample_desired)
    healthy = ActualState(
        model_loaded=True,
        current_model=sample_desired.model,
        container_id="cid-ready",
        health="HEALTHY",
        config_hash=config_hash,
        generation=1,
        gpu_ids="GPU-test",
    )

    with (
        patch.object(ready_reconciler.serving, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch.object(artifacts, "ensure_artifact", return_value="/cache/model"),
        patch.object(ready_reconciler.serving, "stop_if_needed", AsyncMock(return_value=0)),
        patch.object(ready_reconciler.serving, "start_or_update", AsyncMock(return_value="cid-ready")),
        patch.object(ready_reconciler.serving, "wait_for_probes", AsyncMock(return_value=healthy)),
    ):
        await ready_reconciler.reconcile_once()

    app_state, _, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.READY
    desired = await state.get_desired_state()
    assert desired.model == sample_desired.model

    db = await state.get_db()
    async with db.execute("SELECT COUNT(*) FROM reconcile_log") as cur:
        count = (await cur.fetchone())[0]
    assert count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reboot_restores_desired_state_from_sqlite(fresh_state, sample_desired):
    await fresh_state.seed_defaults()
    await fresh_state.update_desired_state(
        DesiredState(model="persisted/model", context_length=4096, gpu_utilization=0.9)
    )
    await fresh_state.set_appliance_state(ApplianceState.READY, last_reconcile_ts=100.0)

    import state as state_mod

    await state_mod.close_db()
    state_mod._db = None

    desired = await state_mod.get_desired_state()
    assert desired.model == "persisted/model"
    assert desired.context_length == 4096


@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_transient_error_stays_reconciling(
    ready_reconciler, sample_desired, patch_reconcile_externals
):
    with (
        patch.object(ready_reconciler.serving, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch.object(artifacts, "ensure_artifact", return_value="/cache/model"),
        patch.object(ready_reconciler.serving, "stop_if_needed", AsyncMock(return_value=0)),
        patch.object(
            ready_reconciler.serving,
            "start_or_update",
            AsyncMock(side_effect=TransientDockerError("409 conflict")),
        ),
        patch.object(ready_reconciler.serving, "heal_environment", AsyncMock(return_value=["removed stale"])),
        patch("reconciler._poll_starting_progress", AsyncMock()),
    ):
        await ready_reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.RECONCILING
    assert "Auto-retry" in (last_error or "")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conflicting_intents_resolve_by_sequence(ready_reconciler):
    await state.append_intent("load_model", {"model": "first/model", "context_length": 1024})
    await state.append_intent("load_model", {"model": "second/model", "context_length": 2048})
    await state.append_intent("load_model", {"model": "third/model", "context_length": 3072})

    with (
        patch.object(ready_reconciler.serving, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch("reconciler.gpu.is_gpu_available", return_value=False),
    ):
        await ready_reconciler.reconcile_once()

    desired = await state.get_desired_state()
    assert desired.model == "third/model"
    assert desired.context_length == 3072


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vram_insufficient_degraded_with_actionable_error(ready_reconciler, sample_desired):
    vram_msg = "GPU VRAM likely insufficient for test"

    with (
        patch.object(ready_reconciler.serving, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch("reconciler.gpu.is_gpu_available", return_value=True),
        patch("reconciler.gpu.check_vram_for_model", return_value=vram_msg),
        patch.object(ready_reconciler.serving, "prune_exited", AsyncMock(return_value=[])),
    ):
        await ready_reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert last_error == vram_msg