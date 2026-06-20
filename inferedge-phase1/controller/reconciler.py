"""Pure reducer reconciler.

Layering Rules
--------------
- This module is the ONLY writer to desired_state, appliance_state, deployments,
  and reconcile_log.
- This module is the ONLY caller of models.py (Docker / vLLM operations).
- The API layer appends intents only; it never imports models.py or calls
  update_desired_state() directly.
"""

import asyncio
import logging
import os
import time

import gpu
import models
import state
from exceptions import (
    ArtifactError,
    DockerError,
    InferEdgeError,
    ProbeTimeoutError,
    TransientArtifactError,
    TransientDockerError,
)
from schemas import ActualState, ApplianceState, ReconcileMetrics

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL = float(os.environ.get("RECONCILE_INTERVAL_SEC", "5"))
HEAL_CHECK_INTERVAL = float(os.environ.get("HEAL_CHECK_INTERVAL_SEC", "30"))
HEAL_STALE_DOWNLOAD_SEC = int(os.environ.get("HEAL_STALE_DOWNLOAD_SEC", "3900"))
HEAL_STALE_STARTING_SEC = int(os.environ.get("HEAL_STALE_STARTING_SEC", "600"))
HEAL_STALE_LOADING_SEC = int(
    os.environ.get("HEAL_STALE_LOADING_SEC", str(int(os.environ.get("VLLM_PROBE_TIMEOUT_SEC", "600")) + 120))
)
ENSURE_ARTIFACT_TIMEOUT = int(
    os.environ.get(
        "ENSURE_ARTIFACT_TIMEOUT_SEC",
        str(int(os.environ.get("HF_DOWNLOAD_TIMEOUT_SEC", "7200")) + 300),
    )
)
VLLM_START_TIMEOUT = int(
    os.environ.get(
        "VLLM_START_TIMEOUT_SEC",
        str(
            int(os.environ.get("VLLM_IMAGE_PULL_TIMEOUT_SEC", "3600"))
            + int(os.environ.get("VLLM_CONTAINER_STARTUP_TIMEOUT_SEC", "120"))
            + 180
        ),
    )
)


async def _poll_starting_progress(model_id: str, stop: asyncio.Event) -> None:
    """Keep /status fresh while vLLM image pull + container create runs."""
    while not stop.is_set():
        pull = await asyncio.to_thread(models.get_vllm_pull_progress)
        if pull:
            progress_msg = (
                f"pulling vLLM image: {pull['percent']:.1f}% ({pull['human']})"
            )
        else:
            progress_msg = "starting vLLM container"
        await state.set_appliance_state(
            ApplianceState.RECONCILING,
            last_error=f"Starting vLLM for {model_id} ({progress_msg})",
            last_reconcile_ts=time.time(),
            actual=ActualState(health="STARTING", current_model=model_id),
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            pass


async def _poll_loading_progress(
    model_id: str, container_id: str, stop: asyncio.Event
) -> None:
    """Keep /status fresh while vLLM loads weights (API may be unreachable)."""
    while not stop.is_set():
        hint = await asyncio.to_thread(models.get_vllm_load_hint, container_id)
        detail = f" ({hint})" if hint else ""
        await state.set_appliance_state(
            ApplianceState.RECONCILING,
            last_error=f"Loading {model_id} into GPU{detail}",
            last_reconcile_ts=time.time(),
            actual=ActualState(
                health="LOADING",
                current_model=model_id,
                container_id=container_id,
            ),
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            pass


async def _poll_download_progress(model_id: str, stop: asyncio.Event) -> None:
    """Update /status while a blocking HF download runs in a worker thread."""
    while not stop.is_set():
        stats = await asyncio.to_thread(models.get_cache_stats, model_id)
        await state.set_appliance_state(
            ApplianceState.RECONCILING,
            last_error=(
                f"Downloading {model_id}: {stats['human']} on disk "
                f"({stats['weight_files']} weight file(s))"
                + (f" — {stats['current_file']}" if stats.get("current_file") else "")
            ),
            last_reconcile_ts=time.time(),
            actual=ActualState(
                health="DOWNLOADING",
                current_model=model_id,
                download_bytes=stats["bytes"],
                download_weight_files=stats["weight_files"],
                download_current_file=stats.get("current_file"),
            ),
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass


class Reconciler:
    def __init__(self, appliance_id: str) -> None:
        self.appliance_id = appliance_id
        self._gate = asyncio.Lock()
        self._busy = False
        self._phase: str | None = None
        self._phase_started_at: float | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self.total_restarts = 0

    def _begin_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_started_at = time.monotonic()

    def _clear_phase(self) -> None:
        self._phase = None
        self._phase_started_at = None

    async def run_loop(self) -> None:
        self._running = True
        watchdog = asyncio.create_task(self._watchdog_loop())
        prewarm = asyncio.create_task(self._prewarm_vllm_image())
        try:
            while self._running:
                try:
                    await self.reconcile_once()
                except Exception:
                    logger.exception("Unhandled error in reconcile loop")
                await asyncio.sleep(RECONCILE_INTERVAL)
        finally:
            watchdog.cancel()
            prewarm.cancel()
            for task in (watchdog, prewarm):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _prewarm_vllm_image(self) -> None:
        try:
            await asyncio.to_thread(models.prewarm_vllm_image)
        except Exception as exc:
            logger.warning("Background vLLM image prewarm failed (will retry): %s", exc)

    async def _watchdog_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HEAL_CHECK_INTERVAL)
            try:
                await self._auto_heal_if_stale()
            except Exception:
                logger.exception("Auto-heal watchdog error")

    def _stale_threshold(self, health: str) -> int:
        return {
            "DOWNLOADING": HEAL_STALE_DOWNLOAD_SEC,
            "STARTING": HEAL_STALE_STARTING_SEC,
            "LOADING": HEAL_STALE_LOADING_SEC,
        }.get(health, HEAL_STALE_STARTING_SEC)

    async def _auto_heal_if_stale(self) -> None:
        app_state, _last_error, last_ts = await state.get_appliance_state()
        if app_state != ApplianceState.RECONCILING:
            return

        actual = await state.get_cached_actual()
        health = actual.health or "UNKNOWN"
        threshold = self._stale_threshold(health)
        stale_sec = time.time() - (last_ts or 0)
        if stale_sec < threshold:
            return

        if self._busy and self._phase_started_at is not None:
            phase_stale = time.monotonic() - self._phase_started_at
            if phase_stale < threshold:
                return

        container_running = await asyncio.to_thread(models.is_vllm_container_running)
        if health == "LOADING" and container_running:
            return

        force_restart = health == "LOADING"
        logger.warning(
            "Auto-heal: stale %s for %.0fs (threshold=%ss, busy=%s, phase=%s)",
            health,
            stale_sec,
            threshold,
            self._busy,
            self._phase,
        )
        actions = await asyncio.to_thread(
            models.heal_deployment_environment, force_restart
        )
        summary = "; ".join(actions) if actions else "no container changes"
        await state.set_appliance_state(
            ApplianceState.RECONCILING,
            last_error=f"Auto-heal: recovered stale {health} after {stale_sec:.0f}s ({summary})",
            last_reconcile_ts=time.time(),
            actual=actual,
        )

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def reconcile_once(self) -> None:
        async with self._gate:
            if self._busy:
                return
            self._busy = True

        try:
            await self._reconcile_body()
        finally:
            async with self._gate:
                self._busy = False
                self._clear_phase()

    async def _reconcile_body(self) -> None:
        start = time.monotonic()
        intents_processed = 0
        vllm_restarts = 0
        action_taken = False
        last_error: str | None = None
        new_state = ApplianceState.RECONCILING
        resolved_model: str | None = None

        try:
            intents_processed = await state.fold_intents_into_desired()
            desired = await state.get_desired_state()
            resolved_model = await state.resolve_model(desired.model)
            desired = desired.model_copy(update={"model": resolved_model})

            config_hash = models.compute_config_hash(desired)
            actual = await models.get_deployment_status(resolved_model)
            deployment = await state.get_deployment_record()

            hash_match = actual.config_hash == config_hash and actual.health == "HEALTHY" and actual.model_loaded

            if hash_match:
                new_state = ApplianceState.READY
                await state.update_desired_state(desired)
                await state.set_appliance_state(
                    new_state, last_error=None, last_reconcile_ts=time.time(), actual=actual
                )
                duration_ms = (time.monotonic() - start) * 1000
                logger.info(
                    "Reconcile NO-OP (ready) duration_ms=%.1f intents=%d",
                    duration_ms,
                    intents_processed,
                )
                return

            action_taken = True
            await state.set_appliance_state(
                ApplianceState.RECONCILING, last_error=None, last_reconcile_ts=time.time()
            )

            if not await asyncio.to_thread(gpu.is_gpu_available):
                last_error = "No GPU detected. Appliance running in CPU-only degraded mode."
                new_state = ApplianceState.DEGRADED
                actual = await models.get_deployment_status(resolved_model)
                await state.update_desired_state(desired)
                await state.set_appliance_state(new_state, last_error, time.time(), actual=actual)
                await state.log_reconcile_event(
                    "gpu_unavailable",
                    {"duration_ms": (time.monotonic() - start) * 1000, "intents_processed": intents_processed},
                )
                return

            self._begin_phase("DOWNLOADING")
            stop_progress = asyncio.Event()
            progress_task = asyncio.create_task(
                _poll_download_progress(resolved_model, stop_progress)
            )
            model_path = ""
            try:
                model_path = await asyncio.wait_for(
                    asyncio.to_thread(models.ensure_artifact, resolved_model),
                    timeout=ENSURE_ARTIFACT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                cache_target = os.path.join(
                    models.CACHE_DIR, models.normalize_model_key(resolved_model)
                )
                await asyncio.to_thread(models.heal_download_environment, cache_target)
                raise TransientArtifactError(
                    f"Model download timed out after {ENSURE_ARTIFACT_TIMEOUT}s; auto-retrying"
                ) from None
            except TransientArtifactError as exc:
                stats = await asyncio.to_thread(models.get_cache_stats, resolved_model)
                await state.set_appliance_state(
                    ApplianceState.RECONCILING,
                    last_error=f"Auto-retry: {exc}",
                    last_reconcile_ts=time.time(),
                    actual=ActualState(
                        health="DOWNLOADING",
                        current_model=resolved_model,
                        download_bytes=stats["bytes"],
                        download_weight_files=stats["weight_files"],
                        download_current_file=stats.get("current_file"),
                    ),
                )
                logger.warning("Transient download issue (will retry): %s", exc)
                return
            except ArtifactError as exc:
                last_error = str(exc)
                new_state = ApplianceState.DEGRADED
                await state.update_desired_state(desired)
                await state.set_appliance_state(
                    new_state, last_error, time.time(), actual=ActualState(health="ARTIFACT_ERROR")
                )
                await state.log_reconcile_event(
                    "artifact_error",
                    {
                        "duration_ms": (time.monotonic() - start) * 1000,
                        "intents_processed": intents_processed,
                        "error": last_error,
                    },
                )
                return
            finally:
                stop_progress.set()
                await progress_task

            self._begin_phase("STARTING")
            await state.set_appliance_state(
                ApplianceState.RECONCILING,
                last_error=f"Starting vLLM for {resolved_model}",
                last_reconcile_ts=time.time(),
                actual=ActualState(health="STARTING", current_model=resolved_model),
            )

            next_gen = await state.get_next_generation()
            stop_starting = asyncio.Event()
            starting_task = asyncio.create_task(
                _poll_starting_progress(resolved_model, stop_starting)
            )
            container_id = ""
            try:
                stopped = await models.stop_vllm_if_needed(
                    except_hash=config_hash, except_generation=next_gen - 1
                )
                if stopped:
                    vllm_restarts += stopped
                    self.total_restarts += stopped

                container_id = await asyncio.wait_for(
                    models.start_or_update_vllm(
                        resolved_model, model_path, desired, config_hash, next_gen
                    ),
                    timeout=VLLM_START_TIMEOUT,
                )
                logger.info("Started vLLM container %s (generation=%d)", container_id[:12], next_gen)
            except asyncio.TimeoutError:
                await asyncio.to_thread(models.heal_deployment_environment, True)
                raise TransientDockerError(
                    f"vLLM start timed out after {VLLM_START_TIMEOUT}s; auto-retrying"
                ) from None
            except TransientDockerError as exc:
                await asyncio.to_thread(models.heal_deployment_environment, True)
                await state.set_appliance_state(
                    ApplianceState.RECONCILING,
                    last_error=f"Auto-retry: {exc}",
                    last_reconcile_ts=time.time(),
                    actual=ActualState(health="STARTING", current_model=resolved_model),
                )
                logger.warning("Transient Docker issue (will retry): %s", exc)
                return
            except DockerError as exc:
                last_error = str(exc)
                new_state = ApplianceState.FAILED
                await state.update_desired_state(desired)
                actual = await models.get_deployment_status(resolved_model)
                await state.set_appliance_state(new_state, last_error, time.time(), actual=actual)
                await state.log_reconcile_event(
                    "docker_error",
                    {
                        "duration_ms": (time.monotonic() - start) * 1000,
                        "intents_processed": intents_processed,
                        "vllm_restarts": vllm_restarts,
                        "error": last_error,
                    },
                )
                return
            finally:
                stop_starting.set()
                await starting_task

            self._begin_phase("LOADING")
            stop_loading = asyncio.Event()
            loading_task = asyncio.create_task(
                _poll_loading_progress(resolved_model, container_id, stop_loading)
            )
            await state.set_appliance_state(
                ApplianceState.RECONCILING,
                last_error=f"Loading {resolved_model} into GPU",
                last_reconcile_ts=time.time(),
                actual=ActualState(
                    health="LOADING",
                    current_model=resolved_model,
                    container_id=container_id,
                ),
            )

            probe_timeout = int(os.environ.get("VLLM_PROBE_TIMEOUT_SEC", "600"))
            try:
                actual = await asyncio.wait_for(
                    models.wait_for_probes(resolved_model),
                    timeout=probe_timeout + 30,
                )
                new_state = ApplianceState.READY
            except asyncio.TimeoutError:
                if await asyncio.to_thread(models.is_vllm_container_running):
                    last_error = (
                        f"Still loading {resolved_model} (API not ready after "
                        f"{probe_timeout}s; container running)"
                    )
                    actual = await models.get_deployment_status(resolved_model)
                    logger.info("%s", last_error)
                else:
                    await asyncio.to_thread(models.heal_deployment_environment, True)
                    last_error = (
                        f"vLLM probes timed out after {probe_timeout}s; auto-retrying"
                    )
                    actual = await models.get_deployment_status(resolved_model)
                    logger.warning("%s", last_error)
                await state.set_appliance_state(
                    ApplianceState.RECONCILING,
                    last_error,
                    time.time(),
                    actual=actual,
                )
                return
            except ProbeTimeoutError as exc:
                if await asyncio.to_thread(models.is_vllm_container_running):
                    last_error = (
                        f"Still loading {resolved_model} (API not ready after "
                        f"{probe_timeout}s; container running)"
                    )
                    actual = await models.get_deployment_status(resolved_model)
                    logger.info("%s", last_error)
                else:
                    await asyncio.to_thread(models.heal_deployment_environment, True)
                    last_error = f"Auto-retry: {exc}"
                    actual = await models.get_deployment_status(resolved_model)
                    logger.warning("Probe timeout (will retry): %s", exc)
                await state.set_appliance_state(
                    ApplianceState.RECONCILING,
                    last_error,
                    time.time(),
                    actual=actual,
                )
                return
            except DockerError as exc:
                last_error = str(exc)
                new_state = ApplianceState.FAILED
                await state.update_desired_state(desired)
                actual = await models.get_deployment_status(resolved_model)
                await state.set_appliance_state(new_state, last_error, time.time(), actual=actual)
                await state.log_reconcile_event(
                    "docker_error",
                    {
                        "duration_ms": (time.monotonic() - start) * 1000,
                        "intents_processed": intents_processed,
                        "vllm_restarts": vllm_restarts,
                        "error": last_error,
                    },
                )
                return
            finally:
                stop_loading.set()
                await loading_task

            await state.update_desired_state(desired)
            await state.set_appliance_state(new_state, last_error, time.time(), actual=actual)

        except TransientArtifactError as exc:
            await state.set_appliance_state(
                ApplianceState.RECONCILING,
                last_error=f"Auto-retry: {exc}",
                last_reconcile_ts=time.time(),
                actual=ActualState(health="DOWNLOADING", current_model=resolved_model),
            )
            logger.warning("Transient download issue (will retry): %s", exc)
            action_taken = True
        except TransientDockerError as exc:
            await asyncio.to_thread(models.heal_deployment_environment, True)
            await state.set_appliance_state(
                ApplianceState.RECONCILING,
                last_error=f"Auto-retry: {exc}",
                last_reconcile_ts=time.time(),
                actual=ActualState(health="STARTING"),
            )
            logger.warning("Transient Docker issue (will retry): %s", exc)
            action_taken = True
        except InferEdgeError as exc:
            last_error = str(exc)
            new_state = ApplianceState.DEGRADED
            await state.set_appliance_state(new_state, last_error, time.time(), actual=ActualState(health="ERROR"))
            action_taken = True
        except Exception as exc:
            last_error = f"Unexpected reconcile error: {exc}"
            new_state = ApplianceState.FAILED
            await state.set_appliance_state(new_state, last_error, time.time(), actual=ActualState(health="ERROR"))
            action_taken = True
            logger.exception("Reconcile failed")

        duration_ms = (time.monotonic() - start) * 1000
        metrics = ReconcileMetrics(
            duration_ms=duration_ms,
            intents_processed=intents_processed,
            vllm_restarts=vllm_restarts,
        )
        logger.info(
            "Reconcile complete state=%s duration_ms=%.1f intents=%d restarts=%d",
            new_state.value,
            duration_ms,
            intents_processed,
            vllm_restarts,
        )

        if action_taken:
            await state.log_reconcile_event(
                f"reconcile_{new_state.value.lower()}",
                metrics.model_dump(),
            )