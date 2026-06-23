"""Regression tests mapped to README acceptance criteria."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import models
import state
from reconciler import Reconciler
from schemas import ActualState, ApplianceState, DesiredState


AC_READY = "Cold boot reaches READY with working LiteLLM endpoint"
AC_STATUS = "/status responsive from t=0 with updating last_reconcile_ts"
AC_REBOOT = "Reboot restores previous model from SQLite desired state"
AC_DOCKER = "Docker daemon restart handled by reconciler recreation"
AC_INTENTS = "Conflicting /models/load calls resolved via intent log ordering"
AC_DEGRADED_CACHE = "Cache/disk errors → DEGRADED with actionable last_error"
AC_CPU = "CPU-only → DEGRADED (not FAILED)"
AC_ONE_VLLM = "Exactly one vLLM container with correct labels"
AC_LITELLM = "End-to-end inference via LiteLLM once READY"
AC_LOG = "reconcile_log grows only on real changes"


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac2_status_responsive_from_boot(fresh_state, env_defaults, monkeypatch):
    """AC2: /status available immediately with BOOT state."""
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "")
    mock_scheduler = MagicMock()
    mock_scheduler.is_ready.return_value = True

    with (
        patch("main.get_scheduler", return_value=mock_scheduler),
        patch("main.Reconciler") as reconciler_cls,
    ):
        reconciler = MagicMock()
        reconciler.run_loop = AsyncMock()
        reconciler.stop = MagicMock()
        reconciler_cls.return_value = reconciler
        from main import app

        with TestClient(app) as client:
            resp = client.get("/status")
            assert resp.status_code == 200
            assert resp.json()["state"] == "BOOT"


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac2_last_reconcile_ts_updates(fresh_state):
    """AC2: last_reconcile_ts updates during reconcile."""
    reconciler = Reconciler("ac-appliance")
    await state.seed_defaults()

    with (
        patch.object(models, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch("reconciler.gpu.is_gpu_available", return_value=False),
    ):
        await reconciler.reconcile_once()

    _, _, ts = await state.get_appliance_state()
    assert ts is not None
    assert ts > 0


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac3_reboot_restores_sqlite_desired_state(fresh_state):
    """AC3: desired state survives DB reconnect (simulated reboot)."""
    await state.seed_defaults()
    await state.update_desired_state(
        DesiredState(model="saved/model", context_length=2048, gpu_utilization=0.95)
    )

    import state as state_mod

    await state_mod.close_db()
    state_mod._db = None

    restored = await state_mod.get_desired_state()
    assert restored.model == "saved/model"
    assert restored.context_length == 2048


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac5_intent_ordering_last_wins(fresh_state):
    """AC5: later sequence_id wins for conflicting load_model intents."""
    await state.seed_defaults()
    for idx, model in enumerate(["alpha", "beta", "gamma"], start=1):
        await state.append_intent("load_model", {"model": f"{model}/model", "context_length": idx * 1024})
    await state.fold_intents_into_desired()
    desired = await state.get_desired_state()
    assert desired.model == "gamma/model"
    assert desired.context_length == 3072


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac6_cache_disk_error_degraded(fresh_state):
    """AC6: permanent artifact errors surface DEGRADED + actionable last_error."""
    from exceptions import ArtifactError

    reconciler = Reconciler("ac-appliance")
    await state.seed_defaults()

    with (
        patch.object(models, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch("reconciler.gpu.is_gpu_available", return_value=True),
        patch("reconciler.gpu.check_vram_for_model", return_value=None),
        patch.object(
            models,
            "ensure_artifact",
            side_effect=ArtifactError("Only 1.0 GB free under /cache; need at least 20 GB"),
        ),
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert "GB free" in (last_error or "")


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac7_cpu_only_degraded_not_failed(fresh_state):
    """AC7: no GPU → DEGRADED, never FAILED."""
    reconciler = Reconciler("ac-appliance")
    await state.seed_defaults()

    with (
        patch.object(models, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch("reconciler.gpu.is_gpu_available", return_value=False),
    ):
        await reconciler.reconcile_once()

    app_state, last_error, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.DEGRADED
    assert app_state != ApplianceState.FAILED
    assert "CPU-only" in (last_error or "")


@pytest.mark.acceptance
def test_ac8_single_vllm_container_labels(mock_docker_client, fake_vllm_container, sample_desired):
    """AC8: managed vLLM identified by required labels."""
    containers = models.find_managed_vllm_containers()
    assert len(containers) == 1
    labels = containers[0].labels
    assert labels["inferedge.managed"] == "true"
    assert labels["inferedge.component"] == "vllm"
    assert labels["inferedge.config_hash"] == models.compute_config_hash(sample_desired)
    assert labels["inferedge.generation"] == "3"
    assert "inferedge.gpu_ids" in labels


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac4_docker_restart_triggers_recreation(
    fresh_state, sample_desired, patch_reconcile_externals
):
    """AC4: missing vLLM after daemon blip triggers start path."""
    reconciler = Reconciler("ac-appliance")
    await state.seed_defaults()
    config_hash = models.compute_config_hash(sample_desired)
    healthy = ActualState(
        model_loaded=True,
        current_model=sample_desired.model,
        container_id="new-cid",
        health="HEALTHY",
        config_hash=config_hash,
        generation=2,
    )

    with (
        patch.object(models, "get_deployment_status", AsyncMock(return_value=ActualState(health="STOPPED"))),
        patch.object(models, "ensure_artifact", return_value="/cache/model"),
        patch.object(models, "stop_vllm_if_needed", AsyncMock(return_value=1)) as stop_mock,
        patch.object(models, "start_or_update_vllm", AsyncMock(return_value="new-cid")) as start_mock,
        patch.object(models, "wait_for_probes", AsyncMock(return_value=healthy)),
    ):
        await reconciler.reconcile_once()

    stop_mock.assert_awaited_once()
    start_mock.assert_awaited_once()
    app_state, _, _ = await state.get_appliance_state()
    assert app_state == ApplianceState.READY


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac10_reconcile_log_no_growth_on_ready_noop(
    fresh_state, sample_desired, patch_reconcile_externals
):
    """AC10: steady READY does not append reconcile_log entries."""
    reconciler = Reconciler("ac-appliance")
    await state.seed_defaults()
    config_hash = models.compute_config_hash(sample_desired)
    healthy = ActualState(
        model_loaded=True,
        current_model=sample_desired.model,
        health="HEALTHY",
        config_hash=config_hash,
    )

    db = await state.get_db()
    async with db.execute("SELECT COUNT(*) FROM reconcile_log") as cur:
        before = (await cur.fetchone())[0]

    with patch.object(models, "get_deployment_status", AsyncMock(return_value=healthy)):
        for _ in range(5):
            await reconciler.reconcile_once()

    async with db.execute("SELECT COUNT(*) FROM reconcile_log") as cur:
        after = (await cur.fetchone())[0]
    assert after == before


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_ac10_vllm_failure_dedup_no_repeat_logs(
    fresh_state, sample_desired, patch_reconcile_externals
):
    """AC10: persistent vLLM load failure does not spam reconcile_log."""
    reconciler = Reconciler("ac-appliance")
    await state.seed_defaults()
    config_hash = models.compute_config_hash(sample_desired)
    await state.update_deployment(config_hash=config_hash, exit_code=1, log_snippet="CUDA error")
    actual = ActualState(health="STOPPED", config_hash=config_hash)

    db = await state.get_db()
    async with db.execute("SELECT COUNT(*) FROM reconcile_log") as cur:
        before = (await cur.fetchone())[0]

    with (
        patch.object(models, "get_deployment_status", AsyncMock(return_value=actual)),
        patch.object(models, "prune_exited_vllm_containers", return_value=[]),
    ):
        for _ in range(3):
            await reconciler.reconcile_once()

    async with db.execute("SELECT COUNT(*) FROM reconcile_log") as cur:
        after = (await cur.fetchone())[0]
    assert after == before