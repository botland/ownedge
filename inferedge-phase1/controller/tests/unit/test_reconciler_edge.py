"""Additional reconciler edge-case coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

import artifacts
import gpu
import state
from exceptions import ProbeTimeoutError
from reconciler import Reconciler
from schemas import ActualState, ApplianceState


@pytest_asyncio.fixture
async def seeded_db(fresh_state):
    await fresh_state.seed_defaults()
    return fresh_state


@pytest.mark.asyncio
async def test_reconcile_probe_timeout_stays_reconciling_when_container_running(
    serving_backend, seeded_db, sample_desired, patch_reconcile_externals
):
    reconciler = Reconciler("edge-appliance", serving_backend=serving_backend)
    actual = ActualState(health="STOPPED")

    with (
        patch.object(
            serving_backend,
            "get_deployment_status",
            AsyncMock(side_effect=[actual, ActualState(health="LOADING", container_id="cid-1")]),
        ),
        patch.object(artifacts, "ensure_artifact", return_value="/cache/model"),
        patch.object(serving_backend, "stop_if_needed", AsyncMock(return_value=0)),
        patch.object(serving_backend, "start_or_update", AsyncMock(return_value="cid-1")),
        patch.object(
            serving_backend,
            "wait_for_probes",
            AsyncMock(side_effect=ProbeTimeoutError("probes timed out")),
        ),
        patch.object(serving_backend, "is_running", AsyncMock(return_value=True)),
        patch("reconciler._poll_starting_progress", AsyncMock()),
        patch("reconciler._poll_loading_progress", AsyncMock()),
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.RECONCILING
    assert "Still loading" in (last_error or "")


@pytest.mark.asyncio
async def test_reconcile_probe_timeout_degraded_when_container_exited(
    serving_backend, seeded_db, sample_desired, patch_reconcile_externals
):
    reconciler = Reconciler("edge-appliance", serving_backend=serving_backend)
    stopped = ActualState(health="STOPPED")
    failed = ActualState(health="STOPPED", exit_code=1, log_snippet="CUDA OOM")

    with (
        patch.object(
            serving_backend,
            "get_deployment_status",
            AsyncMock(side_effect=[stopped, failed, failed]),
        ),
        patch.object(artifacts, "ensure_artifact", return_value="/cache/model"),
        patch.object(serving_backend, "stop_if_needed", AsyncMock(return_value=0)),
        patch.object(serving_backend, "start_or_update", AsyncMock(return_value="cid-1")),
        patch.object(
            serving_backend,
            "wait_for_probes",
            AsyncMock(side_effect=ProbeTimeoutError("probes timed out")),
        ),
        patch.object(serving_backend, "is_running", AsyncMock(return_value=False)),
        patch.object(
            serving_backend,
            "format_load_error",
            MagicMock(return_value="vLLM failed to load model: CUDA OOM"),
        ),
        patch("reconciler._poll_starting_progress", AsyncMock()),
        patch("reconciler._poll_loading_progress", AsyncMock()),
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert "vLLM" in (last_error or "")


@pytest.mark.asyncio
async def test_auto_heal_skips_when_not_reconciling(serving_backend, seeded_db):
    reconciler = Reconciler("edge-appliance", serving_backend=serving_backend)
    await state.set_appliance_state(ApplianceState.READY, last_reconcile_ts=0)
    await reconciler._auto_heal_if_stale()
    app_state, _, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.READY


@pytest.mark.asyncio
async def test_vram_insufficient_logs_and_degrades(
    serving_backend, seeded_db, sample_desired, patch_reconcile_externals
):
    reconciler = Reconciler("edge-appliance", serving_backend=serving_backend)
    msg = "GPU VRAM likely insufficient"

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch.object(gpu, "check_vram_for_model", return_value=msg),
        patch.object(serving_backend, "prune_exited", AsyncMock(return_value=[])),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert last_error == msg
    log_event.assert_called_once()