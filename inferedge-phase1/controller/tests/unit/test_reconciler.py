import time
from unittest.mock import AsyncMock, patch

import artifacts

import pytest
import pytest_asyncio

import state
from exceptions import ArtifactError, DockerError, TransientArtifactError, VllmLoadError
from reconciler import Reconciler
from schemas import ActualState, ApplianceState, DesiredState
from serving.types import compute_config_hash


@pytest_asyncio.fixture
async def seeded_db(fresh_state):
    await fresh_state.seed_defaults()
    return fresh_state


def _matching_healthy_actual(sample_desired):
    config_hash = compute_config_hash(sample_desired)
    return ActualState(
        model_loaded=True,
        current_model=sample_desired.model,
        health="HEALTHY",
        config_hash=config_hash,
    )


@pytest.fixture
def noop_patches(serving_backend):
    """Patches that keep reconcile focused on the code path under test."""
    return {
        "apply_gpu_profile": patch("reconciler.apply_gpu_profile", side_effect=lambda d, m: d),
        "is_gpu_available": patch("reconciler.gpu.is_gpu_available", return_value=True),
        "check_vram": patch("reconciler.gpu.check_vram_for_model", return_value=None),
        "load_hint": patch.object(serving_backend, "get_load_hint", AsyncMock(return_value=None)),
    }


@pytest.mark.asyncio
async def test_reconcile_noop_when_ready(reconciler, serving_backend, seeded_db, sample_desired, noop_patches):
    healthy = _matching_healthy_actual(sample_desired)

    with (
        noop_patches["apply_gpu_profile"],
        noop_patches["is_gpu_available"],
        noop_patches["check_vram"],
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=healthy)),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, _, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.READY
    log_event.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_gpu_unavailable_degraded(reconciler, serving_backend, seeded_db, sample_desired):
    actual = ActualState(health="STOPPED")

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch("reconciler.gpu.is_gpu_available", return_value=False),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert "No GPU detected" in (last_error or "")
    log_event.assert_called_once()
    assert log_event.call_args[0][0] == "gpu_unavailable"


@pytest.mark.asyncio
async def test_reconcile_vllm_load_failure_short_circuit(
    reconciler, serving_backend, seeded_db, sample_desired, noop_patches
):
    config_hash = compute_config_hash(sample_desired)
    actual = ActualState(health="STOPPED", config_hash=config_hash)
    deployment = {
        "config_hash": config_hash,
        "exit_code": 1,
        "log_snippet": "ValueError: KV cache too small",
    }
    await state.update_deployment(**deployment)
    serving_backend.has_load_failure.return_value = True
    serving_backend.format_load_error.return_value = "vLLM failed to load model: KV cache too small"

    with (
        noop_patches["apply_gpu_profile"],
        noop_patches["is_gpu_available"],
        noop_patches["check_vram"],
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch.object(serving_backend, "prune_exited", AsyncMock(return_value=[])),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert "vLLM failed" in (last_error or "")
    log_event.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_artifact_error_degraded(reconciler, serving_backend, seeded_db, sample_desired):
    actual = ActualState(health="STOPPED")

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch("reconciler.gpu.is_gpu_available", return_value=True),
        patch("reconciler.gpu.check_vram_for_model", return_value=None),
        patch.object(artifacts, "ensure_artifact", side_effect=ArtifactError("Disk full under /cache")),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert "Disk full" in (last_error or "")
    log_event.assert_called_once()
    assert log_event.call_args[0][0] == "artifact_error"


@pytest.mark.asyncio
async def test_reconcile_transient_download_retries(reconciler, serving_backend, seeded_db, sample_desired):
    actual = ActualState(health="STOPPED")

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch("reconciler.gpu.is_gpu_available", return_value=True),
        patch("reconciler.gpu.check_vram_for_model", return_value=None),
        patch.object(
            artifacts,
            "ensure_artifact",
            side_effect=TransientArtifactError("Download stalled; retrying"),
        ),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.RECONCILING
    assert "Auto-retry" in (last_error or "")
    log_event.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_happy_path_to_ready(reconciler, serving_backend, seeded_db, sample_desired, noop_patches):
    actual = ActualState(health="STOPPED")
    healthy = _matching_healthy_actual(sample_desired).model_copy(
        update={"container_id": "container-xyz"}
    )

    with (
        noop_patches["apply_gpu_profile"],
        noop_patches["is_gpu_available"],
        noop_patches["check_vram"],
        noop_patches["load_hint"],
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch.object(artifacts, "ensure_artifact", return_value="/models/cache/path"),
        patch.object(serving_backend, "stop_if_needed", AsyncMock(return_value=0)),
        patch.object(serving_backend, "start_or_update", AsyncMock(return_value="container-xyz")),
        patch.object(serving_backend, "wait_for_probes", AsyncMock(return_value=healthy)),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, ts = await state.get_appliance_state()
    assert app_state == ApplianceState.READY
    assert last_error is None
    assert ts is not None
    log_event.assert_called_once()
    assert log_event.call_args[0][0] == "reconcile_ready"


@pytest.mark.asyncio
async def test_reconcile_vllm_load_error_on_start(reconciler, serving_backend, seeded_db, sample_desired):
    actual = ActualState(health="STOPPED")

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch("reconciler.gpu.is_gpu_available", return_value=True),
        patch("reconciler.gpu.check_vram_for_model", return_value=None),
        patch.object(artifacts, "ensure_artifact", return_value="/models/cache/path"),
        patch.object(serving_backend, "stop_if_needed", AsyncMock(return_value=1)),
        patch.object(
            serving_backend,
            "start_or_update",
            AsyncMock(side_effect=VllmLoadError("vLLM failed to load model: CUDA OOM")),
        ),
        patch.object(serving_backend, "heal_environment", AsyncMock(return_value=[])),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert "CUDA OOM" in (last_error or "")
    log_event.assert_called_once()
    assert log_event.call_args[0][0] == "vllm_load_error"


@pytest.mark.asyncio
async def test_reconcile_docker_error_failed(reconciler, serving_backend, seeded_db, sample_desired):
    actual = ActualState(health="STOPPED")

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch("reconciler.gpu.is_gpu_available", return_value=True),
        patch("reconciler.gpu.check_vram_for_model", return_value=None),
        patch.object(artifacts, "ensure_artifact", return_value="/models/cache/path"),
        patch.object(serving_backend, "stop_if_needed", AsyncMock(return_value=0)),
        patch.object(
            serving_backend,
            "start_or_update",
            AsyncMock(side_effect=DockerError("Failed to start vLLM container: denied")),
        ),
        patch.object(serving_backend, "heal_environment", AsyncMock(return_value=[])),
        patch.object(state, "log_reconcile_event", AsyncMock()) as log_event,
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.FAILED
    assert "denied" in (last_error or "")
    log_event.assert_called_once()
    assert log_event.call_args[0][0] == "docker_error"


@pytest.mark.asyncio
async def test_reconcile_processes_intents_before_work(reconciler, serving_backend, seeded_db, sample_desired):
    await state.append_intent(
        "load_model",
        {"model": "casperhansen/llama-3-8b-instruct-awq", "context_length": 2048},
    )
    actual = ActualState(health="STOPPED")

    with (
        patch.object(serving_backend, "get_deployment_status", AsyncMock(return_value=actual)),
        patch("reconciler.gpu.is_gpu_available", return_value=False),
        patch.object(state, "log_reconcile_event", AsyncMock()),
    ):
        await reconciler.reconcile_once()

    desired = await state.get_desired_state()
    assert desired.model == "casperhansen/llama-3-8b-instruct-awq"
    assert desired.context_length == 2048


@pytest.mark.asyncio
async def test_poll_download_progress_updates_status(seeded_db, sample_desired):
    from reconciler import _poll_download_progress

    stop = __import__("asyncio").Event()

    async def stop_soon():
        await __import__("asyncio").sleep(0.05)
        stop.set()

    with patch.object(artifacts, "get_cache_stats", return_value={"bytes": 1024, "human": "1.0 KB", "weight_files": 0}):
        await __import__("asyncio").gather(
            _poll_download_progress(sample_desired.model, stop),
            stop_soon(),
        )

    app_state, last_error, ts = await state.get_appliance_state()
    assert app_state == ApplianceState.RECONCILING
    assert "Downloading" in (last_error or "")
    assert ts is not None