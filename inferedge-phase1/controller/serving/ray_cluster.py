"""Ray Serve LLM serving backend — external cluster deployment."""

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

import state
from compute.ray_cluster import RayClusterScheduler
from exceptions import ProbeTimeoutError
from schemas import ActualState, DesiredState
from serving.base import AbstractServingBackend
from serving.docker_vllm import _detect_quantization, _model_probe_match
from serving.load_errors import format_vllm_load_error, has_vllm_load_failure
from serving.types import normalize_model_key

logger = logging.getLogger(__name__)

RAY_SERVE_BASE_URL = os.environ.get("RAY_SERVE_BASE_URL", "http://localhost:8000").rstrip("/")
RAY_NAMESPACE = os.environ.get("RAY_NAMESPACE", "inferedge")
PROBE_TIMEOUT = int(os.environ.get("VLLM_PROBE_TIMEOUT_SEC", "600"))
SERVE_DEPLOY_TIMEOUT = int(os.environ.get("RAY_SERVE_DEPLOY_TIMEOUT_SEC", "600"))

_active_app_name: str | None = None
_deploy_lock = asyncio.Lock()


def _app_name(config_hash: str) -> str:
    return f"inferedge-{config_hash[:12]}"


def _build_llm_config(
    model_id: str,
    model_path: str,
    desired: DesiredState,
) -> object:
    from ray.serve.llm import LLMConfig

    engine_kwargs: dict = {
        "max_model_len": desired.context_length,
        "gpu_memory_utilization": desired.gpu_utilization,
    }
    quant = _detect_quantization(model_path, model_id)
    if quant:
        engine_kwargs["quantization"] = quant

    return LLMConfig(
        model_loading_config={
            "model_id": model_id,
            "model_source": model_path,
        },
        deployment_config={
            "autoscaling_config": {
                "min_replicas": 1,
                "max_replicas": 1,
            }
        },
        engine_kwargs=engine_kwargs,
    )


def _deploy_sync(
    model_id: str,
    model_path: str,
    desired: DesiredState,
    config_hash: str,
    generation: int,
) -> str:
    from ray import serve
    from ray.serve.llm import build_openai_app

    global _active_app_name
    app_name = _app_name(config_hash)

    if _active_app_name and _active_app_name != app_name:
        try:
            serve.delete(_active_app_name)
            logger.info("Deleted prior Ray Serve app %s", _active_app_name)
        except Exception as exc:
            logger.warning("Could not delete prior Serve app %s: %s", _active_app_name, exc)

    llm_config = _build_llm_config(model_id, model_path, desired)
    app = build_openai_app({"llm_configs": [llm_config]})
    handle = serve.run(
        app,
        name=app_name,
        route_prefix="/",
        blocking=False,
    )
    _active_app_name = app_name
    deployment_id = f"{app_name}-gen{generation}"
    logger.info(
        "Deployed Ray Serve LLM app %s for %s (generation=%d)",
        app_name,
        model_id,
        generation,
    )
    return deployment_id


def _undeploy_sync(except_hash: str | None, except_generation: int | None) -> int:
    from ray import serve

    global _active_app_name
    stopped = 0
    if except_hash:
        keep_name = _app_name(except_hash)
        if _active_app_name == keep_name:
            return 0

    if _active_app_name:
        try:
            serve.delete(_active_app_name)
            stopped = 1
            logger.info("Deleted Ray Serve app %s", _active_app_name)
        except Exception as exc:
            logger.warning("Failed to delete Serve app %s: %s", _active_app_name, exc)
        _active_app_name = None
    return stopped


def _probe_ray_serve(model_id: str) -> ActualState:
    base = RAY_SERVE_BASE_URL
    actual = ActualState(health="STARTING")
    try:
        with httpx.Client(timeout=10.0) as client:
            health_resp = client.get(f"{base}/health")
            if health_resp.status_code != 200:
                actual.health = "UNHEALTHY"
                return actual
            models_resp = client.get(f"{base}/v1/models")
            if models_resp.status_code != 200:
                actual.health = "UNHEALTHY"
                return actual
            data = models_resp.json()
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            model_loaded = _model_probe_match(model_id, model_ids)
            actual.model_loaded = model_loaded
            actual.current_model = model_id if model_loaded else None
            actual.health = "HEALTHY" if model_loaded else "LOADING"
    except httpx.RequestError:
        actual.health = "LOADING" if _active_app_name else "UNREACHABLE"
    return actual


def _wait_for_probes_sync(model_id: str) -> ActualState:
    deadline = time.time() + PROBE_TIMEOUT
    last: ActualState = ActualState(health="STARTING")
    while time.time() < deadline:
        last = _probe_ray_serve(model_id)
        if last.health == "HEALTHY" and last.model_loaded:
            return last
        time.sleep(5)
    raise ProbeTimeoutError(
        f"Ray Serve probes did not pass within {PROBE_TIMEOUT}s (last health={last.health})"
    )


def _get_deployment_status_sync(
    desired_model: Optional[str],
    record: dict,
) -> ActualState:
    if not _active_app_name:
        return ActualState(
            model_loaded=False,
            health="STOPPED",
            config_hash=record.get("config_hash"),
            generation=record.get("generation"),
            gpu_ids=record.get("gpu_ids"),
            exit_code=record.get("exit_code"),
            log_snippet=record.get("log_snippet"),
        )

    actual = ActualState(
        container_id=record.get("container_id"),
        config_hash=record.get("config_hash"),
        generation=record.get("generation"),
        gpu_ids=record.get("gpu_ids"),
        exit_code=record.get("exit_code"),
        log_snippet=record.get("log_snippet"),
        health="RUNNING",
    )
    if desired_model:
        probed = _probe_ray_serve(desired_model)
        actual.model_loaded = probed.model_loaded
        actual.current_model = probed.current_model
        actual.health = probed.health
    return actual


class RayClusterServingBackend(AbstractServingBackend):
    """Ray Serve LLM deployment on an external Ray cluster."""

    def __init__(self, scheduler: RayClusterScheduler) -> None:
        self._scheduler = scheduler

    @property
    def mode(self) -> str:
        return "ray_cluster"

    async def prewarm(self) -> None:
        return None

    async def get_deployment_status(self, desired_model: str | None) -> ActualState:
        record = await state.get_deployment_record()
        return await asyncio.to_thread(_get_deployment_status_sync, desired_model, record)

    async def stop_if_needed(
        self, *, except_hash: str | None, except_generation: int | None
    ) -> int:
        return await asyncio.to_thread(_undeploy_sync, except_hash, except_generation)

    async def start_or_update(
        self,
        model_id: str,
        model_path: str,
        desired: DesiredState,
        config_hash: str,
        generation: int,
    ) -> str:
        async with _deploy_lock:
            deployment_id = await asyncio.to_thread(
                _deploy_sync, model_id, model_path, desired, config_hash, generation
            )
        gpu_ids = ""
        try:
            import ray

            resources = ray.available_resources()
            gpu_count = int(resources.get("GPU", 0))
            if gpu_count:
                gpu_ids = ",".join(f"GPU-{i}" for i in range(gpu_count))
        except Exception:
            pass
        model_key = normalize_model_key(model_id)
        await state.update_deployment(
            container_id=deployment_id,
            config_hash=config_hash,
            generation=generation,
            gpu_ids=gpu_ids or None,
            model_key=model_key,
            exit_code=None,
            log_snippet=None,
        )
        return deployment_id

    async def wait_for_probes(self, model_id: str) -> ActualState:
        return await asyncio.to_thread(_wait_for_probes_sync, model_id)

    async def get_start_progress(self) -> dict:
        return {}

    async def get_load_hint(self, deployment_id: str | None) -> str | None:
        if _active_app_name:
            return f"Ray Serve app {_active_app_name}"
        return None

    async def is_running(self) -> bool:
        return _active_app_name is not None

    async def heal_environment(self, force_restart_running: bool = False) -> list[str]:
        if force_restart_running and _active_app_name:
            stopped = await self.stop_if_needed(except_hash=None, except_generation=None)
            return [f"undeployed {_active_app_name}"] if stopped else []
        return []

    async def prune_exited(self) -> list[str]:
        return []

    def has_load_failure(self, config_hash: str, deployment: dict) -> bool:
        return has_vllm_load_failure(config_hash, deployment)

    def format_load_error(self, record: dict) -> str:
        return format_vllm_load_error(record)