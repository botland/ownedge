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
from exceptions import ArtifactError, DockerError, InferEdgeError, ProbeTimeoutError
from schemas import ActualState, ApplianceState, ReconcileMetrics

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL = float(os.environ.get("RECONCILE_INTERVAL_SEC", "5"))


class Reconciler:
    def __init__(self, appliance_id: str) -> None:
        self.appliance_id = appliance_id
        self.lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._running = False
        self.total_restarts = 0

    async def run_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                await self.reconcile_once()
            except Exception:
                logger.exception("Unhandled error in reconcile loop")
            await asyncio.sleep(RECONCILE_INTERVAL)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def reconcile_once(self) -> None:
        async with self.lock:
            start = time.monotonic()
            intents_processed = 0
            vllm_restarts = 0
            action_taken = False
            last_error: str | None = None
            new_state = ApplianceState.RECONCILING

            try:
                intents_processed = await state.fold_intents_into_desired()
                desired = await state.get_desired_state()
                resolved_model = await state.resolve_model(desired.model)
                desired = desired.model_copy(update={"model": resolved_model})

                config_hash = models.compute_config_hash(desired)
                actual = await models.get_deployment_status(resolved_model)
                deployment = await state.get_deployment_record()
                current_gen = int(deployment.get("generation") or 0)

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

                if not gpu.is_gpu_available():
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

                try:
                    models.ensure_artifact(resolved_model)
                except ArtifactError as exc:
                    last_error = str(exc)
                    new_state = ApplianceState.DEGRADED
                    await state.update_desired_state(desired)
                    await state.set_appliance_state(new_state, last_error, time.time(), actual=ActualState(health="ARTIFACT_ERROR"))
                    await state.log_reconcile_event(
                        "artifact_error",
                        {
                            "duration_ms": (time.monotonic() - start) * 1000,
                            "intents_processed": intents_processed,
                            "error": last_error,
                        },
                    )
                    return

                next_gen = await state.get_next_generation()
                stopped = await models.stop_vllm_if_needed(
                    except_hash=config_hash, except_generation=next_gen - 1
                )
                if stopped:
                    vllm_restarts += stopped
                    self.total_restarts += stopped

                try:
                    container_id = await models.start_or_update_vllm(
                        resolved_model, desired, config_hash, next_gen
                    )
                    logger.info("Started vLLM container %s (generation=%d)", container_id[:12], next_gen)
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

                try:
                    actual = await models.wait_for_probes(resolved_model)
                    new_state = ApplianceState.READY
                except ProbeTimeoutError as exc:
                    last_error = str(exc)
                    actual = await models.get_deployment_status(resolved_model)
                    new_state = ApplianceState.DEGRADED

                await state.update_desired_state(desired)
                await state.set_appliance_state(new_state, last_error, time.time(), actual=actual)

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