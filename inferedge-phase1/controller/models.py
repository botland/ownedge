"""vLLM lifecycle via Docker labels.

IMPORTANT: This module is the sole Docker touchpoint and must only be called
from the reconciler. No API-layer imports.
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

import docker
import httpx
from docker.errors import DockerException
from docker.types import DeviceRequest

import state
from exceptions import ArtifactError, DockerError, ProbeTimeoutError
from gpu import get_gpu_uuids
from schemas import ActualState, DesiredState

logger = logging.getLogger(__name__)

MANAGED_LABEL = "inferedge.managed"
COMPONENT_LABEL = "inferedge.component"
APPLIANCE_LABEL = "inferedge.appliance_id"
MODEL_KEY_LABEL = "inferedge.model_key"
CONFIG_HASH_LABEL = "inferedge.config_hash"
GENERATION_LABEL = "inferedge.generation"
GPU_IDS_LABEL = "inferedge.gpu_ids"

VLLM_ALIAS = "inferedge-vllm"
LOG_TAIL_LINES = 200

APPLIANCE_ID = os.environ.get("APPLIANCE_ID", "inferedge-dev-001")
COMPOSE_PROJECT = os.environ.get("COMPOSE_PROJECT_NAME", "inferedge")
DOCKER_NETWORK = f"{COMPOSE_PROJECT}_default"
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:latest")
VLLM_PORT = int(os.environ.get("VLLM_INTERNAL_PORT", "8000"))
CACHE_DIR = os.environ.get("LOCAL_MODEL_CACHE", "/models_cache")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CONTAINER_STARTUP_TIMEOUT = int(os.environ.get("VLLM_CONTAINER_STARTUP_TIMEOUT_SEC", "120"))
PROBE_TIMEOUT = int(os.environ.get("VLLM_PROBE_TIMEOUT_SEC", "600"))

_docker_client: docker.DockerClient | None = None


def _get_client() -> docker.DockerClient:
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def normalize_model_key(model_id: str) -> str:
    return model_id.replace("/", "--")


def compute_config_hash(desired: DesiredState) -> str:
    payload = json.dumps(
        {
            "model": desired.model,
            "context_length": desired.context_length,
            "gpu_utilization": desired.gpu_utilization,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_labels(model_key: str, config_hash: str, generation: int, gpu_ids: list[str]) -> dict[str, str]:
    return {
        MANAGED_LABEL: "true",
        COMPONENT_LABEL: "vllm",
        APPLIANCE_LABEL: APPLIANCE_ID,
        MODEL_KEY_LABEL: model_key,
        CONFIG_HASH_LABEL: config_hash,
        GENERATION_LABEL: str(generation),
        GPU_IDS_LABEL: ",".join(gpu_ids),
    }


def find_managed_vllm_containers() -> list:
    client = _get_client()
    return client.containers.list(
        all=True,
        filters={"label": f"{MANAGED_LABEL}=true,{COMPONENT_LABEL}=vllm"},
    )


def _capture_container_logs(container, exit_code: int) -> str:
    try:
        logs = container.logs(tail=LOG_TAIL_LINES).decode("utf-8", errors="replace")
    except DockerException as exc:
        logs = f"<failed to read logs: {exc}>"
    return logs[-8000:]


async def _record_unexpected_exit(container, exit_code: int) -> None:
    snippet = _capture_container_logs(container, exit_code)
    labels = container.labels or {}
    await state.update_deployment(
        container_id=container.id,
        config_hash=labels.get(CONFIG_HASH_LABEL),
        generation=int(labels.get(GENERATION_LABEL, "0")),
        gpu_ids=labels.get(GPU_IDS_LABEL),
        model_key=labels.get(MODEL_KEY_LABEL),
        exit_code=exit_code,
        log_snippet=snippet,
    )
    logger.error(
        "vLLM container %s exited unexpectedly (code=%s, generation=%s)",
        container.id[:12],
        exit_code,
        labels.get(GENERATION_LABEL),
    )


def ensure_artifact(model_id: str, cache_dir: str = CACHE_DIR) -> str:
    model_key = normalize_model_key(model_id)
    target = os.path.join(cache_dir, model_key)
    if os.path.isdir(target) and os.listdir(target):
        return target

    os.makedirs(cache_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        token = HF_TOKEN or None
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            token=token,
        )
        return target
    except OSError as exc:
        if exc.errno == 28 or "No space left" in str(exc):
            raise ArtifactError(
                f"Disk full while downloading {model_id}. Free space under {cache_dir}."
            ) from exc
        raise ArtifactError(f"Filesystem error downloading {model_id}: {exc}") from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "403" in msg or "gated" in msg:
            raise ArtifactError(
                f"HF auth failed for {model_id}. Set HF_TOKEN for gated models."
            ) from exc
        if "connection" in msg or "timeout" in msg or "network" in msg:
            raise ArtifactError(f"Network error downloading {model_id}: {exc}") from exc
        raise ArtifactError(f"Failed to download {model_id}: {exc}") from exc


async def stop_vllm_if_needed(
    except_hash: Optional[str] = None,
    except_generation: Optional[int] = None,
) -> int:
    """Stop managed vLLM containers not matching the desired identity. Returns stop count."""
    stopped = 0
    for container in find_managed_vllm_containers():
        labels = container.labels or {}
        config_hash = labels.get(CONFIG_HASH_LABEL)
        generation = int(labels.get(GENERATION_LABEL, "0"))
        if except_hash and config_hash == except_hash:
            if except_generation is None or generation == except_generation:
                if container.status == "running":
                    continue
        try:
            exit_code = container.attrs.get("State", {}).get("ExitCode")
            if container.status != "running" and exit_code not in (None, 0):
                await _record_unexpected_exit(container, int(exit_code))
            container.stop(timeout=30)
            container.remove(force=True)
            stopped += 1
        except DockerException as exc:
            raise DockerError(f"Failed to stop container {container.id[:12]}: {exc}") from exc
    return stopped


async def start_or_update_vllm(
    model_id: str,
    desired: DesiredState,
    config_hash: str,
    generation: int,
) -> str:
    model_key = normalize_model_key(model_id)
    gpu_ids = get_gpu_uuids()
    labels = _build_labels(model_key, config_hash, generation, gpu_ids)

    for container in find_managed_vllm_containers():
        cl = container.labels or {}
        if (
            cl.get(CONFIG_HASH_LABEL) == config_hash
            and int(cl.get(GENERATION_LABEL, "0")) == generation
            and container.status == "running"
        ):
            return container.id

    client = _get_client()
    env = {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {}
    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(desired.context_length),
        "--gpu-memory-utilization",
        str(desired.gpu_utilization),
    ]

    device_requests = [
        DeviceRequest(count=-1, capabilities=[["gpu"]]),
    ]

    try:
        endpoint_config = client.api.create_endpoint_config(aliases=[VLLM_ALIAS])
        networking_config = client.api.create_networking_config(
            {DOCKER_NETWORK: endpoint_config}
        )
        container = client.containers.run(
            VLLM_IMAGE,
            command=command,
            detach=True,
            labels=labels,
            environment=env,
            volumes={CACHE_DIR: {"bind": CACHE_DIR, "mode": "rw"}},
            device_requests=device_requests,
            networking_config=networking_config,
            name=f"inferedge-vllm-gen{generation}",
            remove=False,
        )
    except DockerException as exc:
        raise DockerError(f"Failed to start vLLM container: {exc}") from exc

    deadline = time.time() + CONTAINER_STARTUP_TIMEOUT
    while time.time() < deadline:
        container.reload()
        if container.status == "running":
            break
        if container.status in ("exited", "dead"):
            exit_code = container.attrs.get("State", {}).get("ExitCode", 1)
            await _record_unexpected_exit(container, int(exit_code))
            raise DockerError(f"vLLM container exited during startup (code={exit_code})")
        time.sleep(2)
    else:
        raise DockerError(f"vLLM container did not reach running state within {CONTAINER_STARTUP_TIMEOUT}s")

    await state.update_deployment(
        container_id=container.id,
        config_hash=config_hash,
        generation=generation,
        gpu_ids=",".join(gpu_ids),
        model_key=model_key,
        exit_code=None,
        log_snippet=None,
    )
    return container.id


def _probe_vllm(model_id: str) -> ActualState:
    base = f"http://{VLLM_ALIAS}:{VLLM_PORT}"
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
            model_loaded = any(model_id in mid or mid.endswith(model_id.split("/")[-1]) for mid in model_ids)
            actual.model_loaded = model_loaded
            actual.current_model = model_id if model_loaded else None
            actual.health = "HEALTHY" if model_loaded else "LOADING"
    except httpx.RequestError:
        actual.health = "UNREACHABLE"
    return actual


async def wait_for_probes(model_id: str) -> ActualState:
    deadline = time.time() + PROBE_TIMEOUT
    last: ActualState = ActualState(health="STARTING")
    while time.time() < deadline:
        last = _probe_vllm(model_id)
        if last.health == "HEALTHY" and last.model_loaded:
            return last
        time.sleep(5)
    raise ProbeTimeoutError(
        f"vLLM probes did not pass within {PROBE_TIMEOUT}s (last health={last.health})"
    )


async def get_deployment_status(desired_model: Optional[str] = None) -> ActualState:
    containers = find_managed_vllm_containers()
    running = [c for c in containers if c.status == "running"]

    record = await state.get_deployment_record()

    for container in containers:
        if container.status not in ("running",):
            exit_code = container.attrs.get("State", {}).get("ExitCode")
            if exit_code not in (None, 0):
                await _record_unexpected_exit(container, int(exit_code))

    if not running:
        return ActualState(
            model_loaded=False,
            health="STOPPED",
            config_hash=record.get("config_hash"),
            generation=record.get("generation"),
            gpu_ids=record.get("gpu_ids"),
            exit_code=record.get("exit_code"),
            log_snippet=record.get("log_snippet"),
        )

    if len(running) > 1:
        logger.warning("Multiple vLLM containers running (%d); reporting first", len(running))

    container = running[0]
    labels = container.labels or {}
    actual = ActualState(
        container_id=container.id,
        config_hash=labels.get(CONFIG_HASH_LABEL),
        generation=int(labels.get(GENERATION_LABEL, "0")),
        gpu_ids=labels.get(GPU_IDS_LABEL),
        exit_code=record.get("exit_code"),
        log_snippet=record.get("log_snippet"),
    )

    if desired_model:
        probed = _probe_vllm(desired_model)
        actual.model_loaded = probed.model_loaded
        actual.current_model = probed.current_model
        actual.health = probed.health
    else:
        actual.health = "RUNNING"

    return actual