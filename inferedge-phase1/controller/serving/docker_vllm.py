"""Docker vLLM serving backend — container lifecycle via Docker labels."""

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Optional

import docker
import httpx
from docker.errors import DockerException
from docker.types import DeviceRequest

import state
from artifacts import ensure_model_metadata
from exceptions import (
    DockerError,
    ProbeTimeoutError,
    TransientDockerError,
    VllmLoadError,
)
from gpu import get_gpu_uuids
from schemas import ActualState, DesiredState
from serving.load_errors import format_vllm_load_error
from serving.base import AbstractServingBackend
from serving.types import normalize_model_key

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
# Host path for model cache — required when spawning vLLM via the Docker socket.
MODEL_CACHE_HOST = os.environ.get("MODEL_CACHE_HOST", CACHE_DIR)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DOWNLOAD_TIMEOUT = int(os.environ.get("HF_DOWNLOAD_TIMEOUT_SEC", "7200"))
HF_DOWNLOAD_STALL_SEC = int(os.environ.get("HF_DOWNLOAD_STALL_SEC", "3600"))
HF_API_TIMEOUT = float(os.environ.get("HF_API_TIMEOUT_SEC", "60"))
MIN_FREE_DISK_GB = float(os.environ.get("MIN_FREE_DISK_GB", "20"))
DOWNLOAD_CONNECTIONS = int(os.environ.get("DOWNLOAD_CONNECTIONS", "16"))
_HF_INTERNAL_DIRS = frozenset({".hf_home", ".cache"})
PARALLEL_DOWNLOAD_MIN_BYTES = 50 * 1024 * 1024
CONTAINER_STARTUP_TIMEOUT = int(os.environ.get("VLLM_CONTAINER_STARTUP_TIMEOUT_SEC", "120"))
PROBE_TIMEOUT = int(os.environ.get("VLLM_PROBE_TIMEOUT_SEC", "600"))
VLLM_IMAGE_PULL_TIMEOUT = int(os.environ.get("VLLM_IMAGE_PULL_TIMEOUT_SEC", "3600"))
VLLM_IMAGE_PULL_STALL_SEC = int(os.environ.get("VLLM_IMAGE_PULL_STALL_SEC", "300"))
VLLM_IMAGE_PULL_LOG_MIN_BYTES = int(
    float(os.environ.get("VLLM_IMAGE_PULL_LOG_MIN_BYTES_MB", "50")) * 1024 * 1024
)
VLLM_SHM_SIZE = int(float(os.environ.get("VLLM_SHM_SIZE_GB", "4")) * 1024**3)

_vllm_cli_style_cache: tuple[str, str] | None = None

_docker_client: docker.DockerClient | None = None
_vllm_op_lock = threading.Lock()


def _get_client() -> docker.DockerClient:
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def _vllm_cli_style(client: docker.DockerClient) -> str:
    """Return 'serve' (positional model) or 'flag' (--model) based on image entrypoint."""
    global _vllm_cli_style_cache
    if _vllm_cli_style_cache and _vllm_cli_style_cache[0] == VLLM_IMAGE:
        return _vllm_cli_style_cache[1]
    style = "flag"
    try:
        entrypoint = client.images.get(VLLM_IMAGE).attrs.get("Config", {}).get("Entrypoint") or []
        if entrypoint[:2] == ["vllm", "serve"]:
            style = "serve"
    except DockerException:
        logger.warning("Could not inspect %s entrypoint; defaulting vLLM CLI to --model", VLLM_IMAGE)
    _vllm_cli_style_cache = (VLLM_IMAGE, style)
    return style


def _detect_quantization(model_path: str, model_id: str) -> Optional[str]:
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as config_file:
                config = json.load(config_file)
            quant_cfg = config.get("quantization_config") or {}
            method = quant_cfg.get("quant_method") or quant_cfg.get("quantization_method")
            if method:
                return str(method).lower()
        except (OSError, json.JSONDecodeError):
            pass
    model_lower = model_id.lower()
    for method in ("awq", "gptq", "fp8", "marlin"):
        if method in model_lower:
            return method
    return None

def _build_vllm_command(
    client: docker.DockerClient,
    model_path: str,
    model_id: str,
    desired: DesiredState,
) -> list[str]:
    """Build container args compatible with legacy api_server and modern vllm serve images."""
    common = [
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(desired.context_length),
        "--gpu-memory-utilization",
        str(desired.gpu_utilization),
        "--served-model-name",
        model_id,
    ]
    quant = _detect_quantization(model_path, model_id)
    if quant:
        common.extend(["--quantization", quant])
    if _vllm_cli_style(client) == "serve":
        return [model_path, *common]
    return ["--model", model_path, *common]


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
        filters={
            "label": [f"{MANAGED_LABEL}=true", f"{COMPONENT_LABEL}=vllm"],
        },
    )


def _remove_container_safe(container) -> None:
    """Stop (if running) and remove a container."""
    name = (container.name or container.id[:12]).lstrip("/")
    try:
        container.reload()
        if container.status == "running":
            container.stop(timeout=30)
        container.remove(force=True)
        logger.info("Removed vLLM container %s", name)
    except DockerException as exc:
        raise DockerError(f"Failed to remove container {name}: {exc}") from exc


def _docker_error_is_transient(exc: DockerException) -> bool:
    msg = str(exc).lower()
    return "409" in msg or "conflict" in msg or "already in use" in msg


def _raise_docker_error(action: str, exc: DockerException) -> None:
    if _docker_error_is_transient(exc):
        raise TransientDockerError(f"{action}: {exc}") from exc
    raise DockerError(f"{action}: {exc}") from exc


def _clear_stale_vllm_slot(client: docker.DockerClient, generation: int, target_name: str) -> None:
    """Remove wedged containers occupying a generation name before recreate."""
    for container in find_managed_vllm_containers():
        labels = container.labels or {}
        if int(labels.get(GENERATION_LABEL, "0")) != generation:
            continue
        if container.status == "running":
            continue
        _remove_container_safe(container)

    try:
        existing = client.containers.get(target_name)
    except docker.errors.NotFound:
        return
    if existing.status != "running":
        _remove_container_safe(existing)


def is_vllm_container_running() -> bool:
    return any(c.status == "running" for c in find_managed_vllm_containers())


def get_vllm_load_hint(container_id: str | None = None) -> str | None:
    """Best-effort tail line from the running vLLM container for /status."""
    container = None
    if container_id:
        try:
            candidate = _get_client().containers.get(container_id)
            if candidate.status == "running":
                container = candidate
        except DockerException:
            pass
    if container is None:
        running = [c for c in find_managed_vllm_containers() if c.status == "running"]
        if not running:
            return None
        container = running[0]
    try:
        raw = container.logs(tail=40).decode("utf-8", errors="replace")
    except DockerException:
        return None
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        if stripped.startswith("\x1b"):
            continue
        return stripped[-200:]
    return None


def _capture_container_logs(container, exit_code: int) -> str:
    try:
        logs = container.logs(tail=LOG_TAIL_LINES).decode("utf-8", errors="replace")
    except DockerException as exc:
        logs = f"<failed to read logs: {exc}>"
    return logs[-8000:]



def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.2f} GB"


def _exit_record_from_container(container, exit_code: int) -> dict:
    labels = container.labels or {}
    return {
        "container_id": container.id,
        "config_hash": labels.get(CONFIG_HASH_LABEL),
        "generation": int(labels.get(GENERATION_LABEL, "0")),
        "gpu_ids": labels.get(GPU_IDS_LABEL),
        "model_key": labels.get(MODEL_KEY_LABEL),
        "exit_code": exit_code,
        "log_snippet": _capture_container_logs(container, exit_code),
    }


def _exit_record_changed(stored: dict, record: dict) -> bool:
    """True when this exit record is new or materially different from SQLite."""
    for key in ("container_id", "exit_code", "generation", "config_hash", "log_snippet"):
        if stored.get(key) != record.get(key):
            return True
    return False


async def _apply_exit_records(records: list[dict]) -> None:
    stored = await state.get_deployment_record()
    for record in records:
        changed = _exit_record_changed(stored, record)
        await state.update_deployment(**record)
        stored = await state.get_deployment_record()
        if changed:
            logger.error(
                "vLLM container %s exited unexpectedly (code=%s, generation=%s)",
                record["container_id"][:12],
                record["exit_code"],
                record.get("generation"),
            )


def prune_exited_vllm_containers() -> list[str]:
    """Remove non-running managed vLLM containers (idempotent auto-heal)."""
    actions: list[str] = []
    for container in find_managed_vllm_containers():
        if container.status == "running":
            continue
        name = (container.name or container.id[:12]).lstrip("/")
        try:
            _remove_container_safe(container)
            actions.append(f"removed {name} ({container.status})")
        except DockerException as exc:
            actions.append(f"failed to remove {name}: {exc}")
    if actions:
        logger.info("Pruned exited vLLM containers: %s", "; ".join(actions))
    return actions


def _stop_vllm_if_needed_sync(
    except_hash: Optional[str] = None,
    except_generation: Optional[int] = None,
) -> tuple[int, list[dict]]:
    """Stop managed vLLM containers not matching the desired identity."""
    stopped = 0
    exit_records: list[dict] = []
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
                exit_records.append(_exit_record_from_container(container, int(exit_code)))
            _remove_container_safe(container)
            stopped += 1
        except DockerException as exc:
            _raise_docker_error(f"Failed to stop container {container.id[:12]}", exc)
    return stopped, exit_records


async def stop_vllm_if_needed(
    except_hash: Optional[str] = None,
    except_generation: Optional[int] = None,
) -> int:
    stopped, exit_records = await asyncio.to_thread(
        _stop_vllm_if_needed_sync, except_hash, except_generation
    )
    await _apply_exit_records(exit_records)
    return stopped


def _vllm_image_present(client: docker.DockerClient) -> bool:
    try:
        client.images.get(VLLM_IMAGE)
        return True
    except docker.errors.ImageNotFound:
        return False


_VLLM_PULL_PROGRESS_FILE = os.path.join(CACHE_DIR, ".inferedge-vllm-pull-progress")
_VLLM_PULL_TRACKED_STATUSES = frozenset({
    "Pulling fs layer",
    "Downloading",
    "Extracting",
    "Download complete",
    "Pull complete",
    "Waiting",
    "Pulling from",
})


def _vllm_pull_progress_path() -> str:
    return _VLLM_PULL_PROGRESS_FILE


def get_vllm_pull_progress() -> dict:
    """Return in-progress vLLM image pull stats for /status polling."""
    path = _vllm_pull_progress_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as progress_f:
            raw = progress_f.read().strip()
    except OSError:
        return {}
    if not raw:
        return {}
    percent_str, _, bytes_summary = raw.partition("|")
    try:
        percent = float(percent_str)
    except ValueError:
        return {}
    return {
        "percent": percent,
        "human": bytes_summary or f"{percent:.1f}%",
    }


def _clear_vllm_pull_progress() -> None:
    try:
        os.remove(_vllm_pull_progress_path())
    except OSError:
        pass


def _write_vllm_pull_progress(percent: float | None, bytes_summary: str | None = None) -> None:
    if percent is None:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        content = f"{percent:.1f}"
        if bytes_summary:
            content += f"|{bytes_summary}"
        with open(_vllm_pull_progress_path(), "w", encoding="utf-8") as progress_f:
            progress_f.write(content)
    except OSError:
        pass


def _docker_pull_progress_bytes(line: dict) -> tuple[int | None, int | None]:
    detail = line.get("progressDetail") or {}
    current = detail.get("current")
    total = detail.get("total")
    if not isinstance(total, int) or total <= 0:
        if isinstance(current, int):
            return current, None
        return None, None
    current_val = int(current) if isinstance(current, int) else 0
    return current_val, int(total)


def _update_docker_pull_layers(
    layer_id: str,
    status: str,
    line: dict,
    layer_totals: dict[str, int],
    layer_current: dict[str, int],
) -> None:
    if not layer_id:
        return
    current, total = _docker_pull_progress_bytes(line)
    if total is not None:
        layer_totals[layer_id] = total
    if current is not None:
        layer_current[layer_id] = current
    if status in {"Download complete", "Pull complete"} and layer_id in layer_totals:
        layer_current[layer_id] = layer_totals[layer_id]


def _docker_pull_done_bytes(
    layer_totals: dict[str, int],
    layer_current: dict[str, int],
) -> int:
    return sum(
        min(layer_current.get(layer_id, 0), layer_total)
        for layer_id, layer_total in layer_totals.items()
    )


def _docker_pull_overall_percent(
    layer_totals: dict[str, int],
    layer_current: dict[str, int],
) -> float | None:
    if not layer_totals:
        return None
    total_bytes = sum(layer_totals.values())
    if total_bytes <= 0:
        return None
    done_bytes = _docker_pull_done_bytes(layer_totals, layer_current)
    return min(100.0, done_bytes * 100.0 / total_bytes)


def _docker_pull_bytes_summary(
    layer_totals: dict[str, int],
    layer_current: dict[str, int],
) -> str | None:
    if not layer_totals:
        return None
    total_bytes = sum(layer_totals.values())
    if total_bytes <= 0:
        return None
    done_bytes = _docker_pull_done_bytes(layer_totals, layer_current)
    return f"{_format_bytes(done_bytes)}/{_format_bytes(total_bytes)}"


def _format_vllm_pull_log(
    status: str,
    layer_id: str,
    progress: str,
    overall_pct: float | None,
    bytes_summary: str | None,
) -> str:
    layer_ref = layer_id[:12] if layer_id else ""
    detail = " ".join(part for part in (layer_ref, status, progress) if part)
    if overall_pct is not None:
        headline = f"{overall_pct:.1f}%"
        if bytes_summary:
            headline = f"{headline} ({bytes_summary})"
        return f"vLLM pull: {headline} — {detail}" if detail else f"vLLM pull: {headline}"
    return f"vLLM pull: {detail}" if detail else f"vLLM pull: {status}"


def _should_log_vllm_pull_progress(
    done_bytes: int,
    last_logged_done_bytes: int,
    overall_pct: float | None,
    last_logged_pct: float | None,
) -> bool:
    if last_logged_done_bytes < 0:
        return True
    if done_bytes - last_logged_done_bytes >= VLLM_IMAGE_PULL_LOG_MIN_BYTES:
        return True
    if overall_pct is not None and last_logged_pct is not None:
        return round(overall_pct, 1) != round(last_logged_pct, 1)
    return done_bytes != last_logged_done_bytes


def _pull_vllm_image_worker(image: str, pull_stall_sec: int) -> None:
    """Pull vLLM image in an isolated subprocess (killable on timeout/stall)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = docker.from_env()
    logger.info("Pulling vLLM image %s", image)
    _clear_vllm_pull_progress()
    last_stream_activity = time.monotonic()
    last_byte_activity = time.monotonic()
    last_done_bytes = 0
    last_logged_done_bytes = -1
    last_logged_pct: float | None = None
    layer_totals: dict[str, int] = {}
    layer_current: dict[str, int] = {}
    try:
        for line in client.api.pull(image, stream=True, decode=True):
            now = time.monotonic()
            if now - last_stream_activity > pull_stall_sec:
                raise RuntimeError(f"Stalled pulling {image} for {pull_stall_sec}s (no Docker events)")
            status = line.get("status") or ""
            layer_id = line.get("id") or ""
            progress = line.get("progress") or ""
            error = line.get("error")
            if error:
                raise RuntimeError(f"Failed to pull {image}: {error}")
            if status == "Pulling fs layer":
                last_stream_activity = now
                continue
            if status in _VLLM_PULL_TRACKED_STATUSES:
                last_stream_activity = now
                _update_docker_pull_layers(layer_id, status, line, layer_totals, layer_current)
                done_bytes = _docker_pull_done_bytes(layer_totals, layer_current)
                if done_bytes > last_done_bytes:
                    last_done_bytes = done_bytes
                    last_byte_activity = now
                elif (
                    status == "Downloading"
                    and done_bytes > 0
                    and now - last_byte_activity > pull_stall_sec
                ):
                    summary = _docker_pull_bytes_summary(layer_totals, layer_current)
                    raise RuntimeError(
                        f"Stalled pulling {image} for {pull_stall_sec}s at {summary or 'unknown progress'}"
                    )
                overall_pct = _docker_pull_overall_percent(layer_totals, layer_current)
                bytes_summary = _docker_pull_bytes_summary(layer_totals, layer_current)
                _write_vllm_pull_progress(overall_pct, bytes_summary)
                if status in {"Downloading", "Extracting"}:
                    if not _should_log_vllm_pull_progress(
                        done_bytes, last_logged_done_bytes, overall_pct, last_logged_pct
                    ):
                        continue
                    last_logged_done_bytes = done_bytes
                    last_logged_pct = overall_pct
                logger.info(
                    _format_vllm_pull_log(status, layer_id, progress, overall_pct, bytes_summary)
                )
    except DockerException as exc:
        raise RuntimeError(f"Failed to pull {image}: {exc}") from exc
    client.images.get(image)
    _clear_vllm_pull_progress()
    logger.info("vLLM image pull finished: %s", image)


def _run_vllm_image_pull_watchdog() -> None:
    """Run image pull in a child process; kill on timeout or stall."""
    import multiprocessing as mp

    client = _get_client()
    if _vllm_image_present(client):
        logger.info("vLLM image present: %s", VLLM_IMAGE)
        return

    _clear_vllm_pull_progress()
    logger.info(
        "Pulling vLLM image %s (timeout=%ss, stall=%ss)",
        VLLM_IMAGE,
        VLLM_IMAGE_PULL_TIMEOUT,
        VLLM_IMAGE_PULL_STALL_SEC,
    )
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_pull_vllm_image_worker,
        args=(VLLM_IMAGE, VLLM_IMAGE_PULL_STALL_SEC),
        daemon=True,
    )
    proc.start()
    deadline = time.monotonic() + VLLM_IMAGE_PULL_TIMEOUT
    while proc.is_alive():
        if time.monotonic() > deadline:
            proc.terminate()
            proc.join(timeout=15)
            raise TransientDockerError(
                f"Timed out pulling {VLLM_IMAGE} after {VLLM_IMAGE_PULL_TIMEOUT}s"
            )
        proc.join(timeout=10)
    if proc.exitcode != 0:
        raise TransientDockerError(f"vLLM image pull failed for {VLLM_IMAGE} (exit {proc.exitcode})")


def _ensure_vllm_image(client: docker.DockerClient) -> None:
    """Pull vLLM image with visible progress — containers.run() pulls silently otherwise."""
    if _vllm_image_present(client):
        return
    _run_vllm_image_pull_watchdog()
    if not _vllm_image_present(client):
        raise TransientDockerError(f"vLLM image {VLLM_IMAGE} missing after pull")


def prewarm_vllm_image() -> None:
    """Background pull so first reconcile does not block on a silent image fetch."""
    try:
        _ensure_vllm_image(_get_client())
    except TransientDockerError as exc:
        logger.warning("vLLM image prewarm incomplete (will retry): %s", exc)


def heal_deployment_environment(force_restart_running: bool = False) -> list[str]:
    """Auto-heal wedged vLLM containers blocking the next reconcile attempt."""
    actions: list[str] = []
    client = _get_client()
    for container in find_managed_vllm_containers():
        status = container.status
        if status == "running" and not force_restart_running:
            continue
        name = (container.name or container.id[:12]).lstrip("/")
        try:
            _remove_container_safe(container)
            actions.append(f"removed {name} ({status})")
        except DockerException as exc:
            actions.append(f"failed to remove {name}: {exc}")

    for container in client.containers.list(all=True):
        name = (container.name or "").lstrip("/")
        if not name.startswith("inferedge-vllm-gen"):
            continue
        labels = container.labels or {}
        if labels.get(MANAGED_LABEL) == "true":
            continue
        try:
            _remove_container_safe(container)
            actions.append(f"removed orphan {name}")
        except DockerException as exc:
            actions.append(f"failed to remove orphan {name}: {exc}")

    if actions:
        logger.info("Auto-heal deployment: %s", "; ".join(actions))
    return actions


def _start_or_update_vllm_sync(
    model_id: str,
    model_path: str,
    desired: DesiredState,
    config_hash: str,
    generation: int,
) -> tuple[str, Optional[dict]]:
    if not _vllm_op_lock.acquire(blocking=False):
        raise TransientDockerError("vLLM start already in progress; will retry")
    try:
        return _start_or_update_vllm_sync_locked(
            model_id, model_path, desired, config_hash, generation
        )
    finally:
        _vllm_op_lock.release()


def _start_or_update_vllm_sync_locked(
    model_id: str,
    model_path: str,
    desired: DesiredState,
    config_hash: str,
    generation: int,
) -> tuple[str, Optional[dict]]:
    model_key = normalize_model_key(model_id)
    target_name = f"inferedge-vllm-gen{generation}"
    logger.info("Starting vLLM for %s (generation=%d)", model_id, generation)
    gpu_ids = get_gpu_uuids()
    labels = _build_labels(model_key, config_hash, generation, gpu_ids)

    client = _get_client()
    for container in find_managed_vllm_containers():
        cl = container.labels or {}
        if (
            cl.get(CONFIG_HASH_LABEL) == config_hash
            and int(cl.get(GENERATION_LABEL, "0")) == generation
            and container.status == "running"
        ):
            logger.info("Reusing running vLLM container %s", container.id[:12])
            return container.id, None

    _clear_stale_vllm_slot(client, generation, target_name)
    _ensure_vllm_image(client)
    ensure_model_metadata(model_id, model_path)
    env = {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {}
    command = _build_vllm_command(client, model_path, model_id, desired)

    device_requests = [
        DeviceRequest(count=-1, capabilities=[["gpu"]]),
    ]

    logger.info(
        "Creating vLLM container %s on network %s (shm=%s, cache=%s:%s)",
        target_name,
        DOCKER_NETWORK,
        _format_bytes(VLLM_SHM_SIZE),
        MODEL_CACHE_HOST,
        CACHE_DIR,
    )
    container = None
    try:
        container = client.containers.run(
            VLLM_IMAGE,
            command=command,
            detach=True,
            labels=labels,
            environment=env,
            volumes={MODEL_CACHE_HOST: {"bind": CACHE_DIR, "mode": "rw"}},
            device_requests=device_requests,
            network=DOCKER_NETWORK,
            name=target_name,
            remove=False,
            shm_size=VLLM_SHM_SIZE,
        )
        network = client.networks.get(DOCKER_NETWORK)
        try:
            network.disconnect(container.id, force=True)
        except DockerException:
            pass
        network.connect(container.id, aliases=[VLLM_ALIAS])
    except DockerException as exc:
        _clear_stale_vllm_slot(client, generation, target_name)
        _raise_docker_error("Failed to start vLLM container", exc)

    deadline = time.time() + CONTAINER_STARTUP_TIMEOUT
    while time.time() < deadline:
        container.reload()
        if container.status == "running":
            break
        if container.status in ("exited", "dead"):
            exit_code = int(container.attrs.get("State", {}).get("ExitCode", 1))
            return "", _exit_record_from_container(container, exit_code)
        time.sleep(2)
    else:
        raise DockerError(f"vLLM container did not reach running state within {CONTAINER_STARTUP_TIMEOUT}s")

    return container.id, None


async def start_or_update_vllm(
    model_id: str,
    model_path: str,
    desired: DesiredState,
    config_hash: str,
    generation: int,
) -> str:
    container_id, exit_record = await asyncio.to_thread(
        _start_or_update_vllm_sync, model_id, model_path, desired, config_hash, generation
    )
    if exit_record:
        await _apply_exit_records([exit_record])
        raise VllmLoadError(format_vllm_load_error(exit_record))

    gpu_ids = get_gpu_uuids()
    model_key = normalize_model_key(model_id)
    await state.update_deployment(
        container_id=container_id,
        config_hash=config_hash,
        generation=generation,
        gpu_ids=",".join(gpu_ids),
        model_key=model_key,
        exit_code=None,
        log_snippet=None,
    )
    return container_id


def _model_probe_match(model_id: str, served_ids: list[str]) -> bool:
    key = normalize_model_key(model_id)
    tail = model_id.split("/")[-1]
    for mid in served_ids:
        if not mid:
            continue
        normalized = mid.rstrip("/")
        if (
            model_id in mid
            or key in mid
            or normalized.endswith(tail)
            or normalized.endswith(key)
        ):
            return True
    return False


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
            model_loaded = _model_probe_match(model_id, model_ids)
            actual.model_loaded = model_loaded
            actual.current_model = model_id if model_loaded else None
            actual.health = "HEALTHY" if model_loaded else "LOADING"
    except httpx.RequestError:
        # vLLM does not bind :8000 until weights are loaded; a running container
        # with no listener is still loading, not a network failure.
        actual.health = "LOADING" if is_vllm_container_running() else "UNREACHABLE"
    return actual


def _raise_if_vllm_container_exited() -> None:
    for container in find_managed_vllm_containers():
        if container.status in ("exited", "dead"):
            exit_code = int(container.attrs.get("State", {}).get("ExitCode", 1))
            record = _exit_record_from_container(container, exit_code)
            raise VllmLoadError(format_vllm_load_error(record))


def _wait_for_probes_sync(model_id: str) -> ActualState:
    deadline = time.time() + PROBE_TIMEOUT
    last: ActualState = ActualState(health="STARTING")
    while time.time() < deadline:
        _raise_if_vllm_container_exited()
        last = _probe_vllm(model_id)
        if last.health == "HEALTHY" and last.model_loaded:
            return last
        time.sleep(5)
    raise ProbeTimeoutError(
        f"vLLM probes did not pass within {PROBE_TIMEOUT}s (last health={last.health})"
    )


async def wait_for_probes(model_id: str) -> ActualState:
    return await asyncio.to_thread(_wait_for_probes_sync, model_id)


def _get_deployment_status_sync(
    desired_model: Optional[str],
    record: dict,
) -> tuple[ActualState, list[dict]]:
    containers = find_managed_vllm_containers()
    running = [c for c in containers if c.status == "running"]
    exit_records: list[dict] = []

    latest_failed: tuple[int, dict] | None = None
    for container in containers:
        if container.status not in ("running",):
            exit_code = container.attrs.get("State", {}).get("ExitCode")
            if exit_code in (None, 0):
                continue
            labels = container.labels or {}
            generation = int(labels.get(GENERATION_LABEL, "0"))
            failed = _exit_record_from_container(container, int(exit_code))
            if latest_failed is None or generation >= latest_failed[0]:
                latest_failed = (generation, failed)
    exit_records = [latest_failed[1]] if latest_failed else []

    if not running:
        return ActualState(
            model_loaded=False,
            health="STOPPED",
            config_hash=record.get("config_hash"),
            generation=record.get("generation"),
            gpu_ids=record.get("gpu_ids"),
            exit_code=record.get("exit_code"),
            log_snippet=record.get("log_snippet"),
        ), exit_records

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

    return actual, exit_records


async def get_deployment_status(desired_model: Optional[str] = None) -> ActualState:
    record = await state.get_deployment_record()
    actual, exit_records = await asyncio.to_thread(
        _get_deployment_status_sync, desired_model, record
    )
    await _apply_exit_records(exit_records)
    return actual


class DockerVllmServingBackend(AbstractServingBackend):
    """Docker-managed vLLM runtime (litellm_vllm mode)."""

    @property
    def mode(self) -> str:
        return "litellm_vllm"

    async def prewarm(self) -> None:
        await asyncio.to_thread(prewarm_vllm_image)

    async def get_deployment_status(self, desired_model: str | None) -> ActualState:
        return await get_deployment_status(desired_model)

    async def stop_if_needed(
        self, *, except_hash: str | None, except_generation: int | None
    ) -> int:
        return await stop_vllm_if_needed(
            except_hash=except_hash, except_generation=except_generation
        )

    async def start_or_update(
        self,
        model_id: str,
        model_path: str,
        desired: DesiredState,
        config_hash: str,
        generation: int,
    ) -> str:
        return await start_or_update_vllm(
            model_id, model_path, desired, config_hash, generation
        )

    async def wait_for_probes(self, model_id: str) -> ActualState:
        return await wait_for_probes(model_id)

    async def get_start_progress(self) -> dict:
        return await asyncio.to_thread(get_vllm_pull_progress)

    async def get_load_hint(self, deployment_id: str | None) -> str | None:
        return await asyncio.to_thread(get_vllm_load_hint, deployment_id)

    async def is_running(self) -> bool:
        return await asyncio.to_thread(is_vllm_container_running)

    async def heal_environment(self, force_restart_running: bool = False) -> list[str]:
        return await asyncio.to_thread(heal_deployment_environment, force_restart_running)

    async def prune_exited(self) -> list[str]:
        return await asyncio.to_thread(prune_exited_vllm_containers)

    def has_load_failure(self, config_hash: str, deployment: dict) -> bool:
        from serving.load_errors import has_vllm_load_failure

        return has_vllm_load_failure(config_hash, deployment)

    def format_load_error(self, record: dict) -> str:
        return format_vllm_load_error(record)