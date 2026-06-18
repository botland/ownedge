"""vLLM lifecycle via Docker labels.

IMPORTANT: This module is the sole Docker touchpoint and must only be called
from the reconciler. No API-layer imports.
"""

import asyncio
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
HF_DOWNLOAD_TIMEOUT = int(os.environ.get("HF_DOWNLOAD_TIMEOUT_SEC", "7200"))
HF_DOWNLOAD_STALL_SEC = int(os.environ.get("HF_DOWNLOAD_STALL_SEC", "1200"))
MIN_FREE_DISK_GB = float(os.environ.get("MIN_FREE_DISK_GB", "20"))
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


GATED_MODEL_PREFIXES = ("meta-llama/", "google/gemma-")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _hf_auth_error(model_id: str) -> ArtifactError:
    return ArtifactError(
        f"HF auth required for {model_id}. "
        f"Set HF_TOKEN in .env (https://huggingface.co/settings/tokens), "
        f"then accept the license at https://huggingface.co/{model_id}"
    )


def _hf_license_error(model_id: str) -> ArtifactError:
    return ArtifactError(
        f"HF access denied (403) for {model_id}. HF_TOKEN is set but this account "
        f"has not been granted access. Log into https://huggingface.co as the token owner, "
        f"open https://huggingface.co/{model_id}, and click 'Agree and access repository'."
    )


def _is_likely_gated(model_id: str) -> bool:
    return any(model_id.startswith(prefix) for prefix in GATED_MODEL_PREFIXES)


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.2f} GB"


def get_cache_stats(model_id: str, cache_dir: str = CACHE_DIR) -> dict:
    """Return on-disk download progress for a model cache directory."""
    target = os.path.join(cache_dir, normalize_model_key(model_id))
    total_bytes = 0
    weight_files = 0
    current_file = None
    progress_path = os.path.join(target, ".inferedge-download-progress")
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, encoding="utf-8") as progress_f:
                current_file = progress_f.read().strip()
        except OSError:
            pass
    if os.path.isdir(target):
        for root, _, files in os.walk(target):
            for name in files:
                if name == ".inferedge-download-progress":
                    continue
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                total_bytes += size
                if name.endswith(WEIGHT_SUFFIXES) and ".incomplete" not in name:
                    weight_files += 1
    return {
        "path": target,
        "bytes": total_bytes,
        "human": _format_bytes(total_bytes),
        "weight_files": weight_files,
        "current_file": current_file,
    }


def _cache_has_weights(target: str) -> bool:
    if not os.path.isdir(target):
        return False
    for root, _, files in os.walk(target):
        for name in files:
            if name.endswith(WEIGHT_SUFFIXES):
                return True
    return False


def _clear_cache_dir(target: str) -> None:
    import shutil

    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)


def _raise_for_hf_http(model_id: str, exc: Exception) -> None:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if isinstance(exc, GatedRepoError):
        raise (_hf_license_error(model_id) if HF_TOKEN else _hf_auth_error(model_id)) from exc
    if isinstance(exc, HfHubHTTPError) and exc.response is not None:
        code = exc.response.status_code
        if code == 401:
            raise _hf_auth_error(model_id) from exc
        if code == 403:
            raise (_hf_license_error(model_id) if HF_TOKEN else _hf_auth_error(model_id)) from exc


def _should_download_file(filename: str) -> bool:
    if filename.endswith(".gguf") or filename.startswith("original/"):
        return False
    if filename.endswith((".safetensors", ".bin", ".json", ".txt", ".jinja", ".model")):
        return True
    return "tokenizer" in filename.lower()


def _list_repo_files(model_id: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN or None)
    return [f for f in api.list_repo_files(model_id) if _should_download_file(f)]


def _check_disk_space(cache_dir: str) -> None:
    import shutil

    usage = shutil.disk_usage(cache_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < MIN_FREE_DISK_GB:
        raise ArtifactError(
            f"Only {free_gb:.1f} GB free under {cache_dir}; "
            f"need at least {MIN_FREE_DISK_GB:.0f} GB for an 8B model download."
        )


def _write_download_progress(target: str, message: str) -> None:
    try:
        with open(os.path.join(target, ".inferedge-download-progress"), "w", encoding="utf-8") as f:
            f.write(message)
    except OSError:
        pass


def _download_all_files(model_id: str, target: str) -> None:
    """Download HF repo files one at a time (runs in subprocess for killability)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    # Keep HF hub cache on the model volume so partial downloads are visible in /status
    hf_home = os.path.join(target, ".hf_home")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

    files = _list_repo_files(model_id)
    if not files:
        raise ArtifactError(f"No downloadable vLLM files found in {model_id}")

    os.makedirs(target, exist_ok=True)
    total = len(files)
    for idx, filename in enumerate(files, 1):
        progress = f"[{idx}/{total}] {filename}"
        _write_download_progress(target, progress)
        logger.info("Downloading %s %s", model_id, progress)
        try:
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=target,
                token=HF_TOKEN or None,
            )
        except (GatedRepoError, HfHubHTTPError) as exc:
            _raise_for_hf_http(model_id, exc)
        except Exception as exc:
            err_file = os.path.join(target, ".inferedge-download-error")
            try:
                with open(err_file, "w", encoding="utf-8") as f:
                    f.write(str(exc))
            except OSError:
                pass
            raise
    _write_download_progress(target, "complete")


def _run_download_with_watchdog(model_id: str, target: str, cache_dir: str) -> None:
    """Run download in a child process; kill on timeout or stall."""
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    proc = ctx.Process(target=_download_all_files, args=(model_id, target), daemon=True)
    proc.start()

    last_bytes = 0
    last_progress = None
    stall_since = time.monotonic()
    deadline = time.monotonic() + HF_DOWNLOAD_TIMEOUT

    while proc.is_alive():
        if time.monotonic() > deadline:
            proc.terminate()
            proc.join(timeout=15)
            raise ArtifactError(
                f"Download timed out after {HF_DOWNLOAD_TIMEOUT}s for {model_id}"
            )

        stats = get_cache_stats(model_id, cache_dir)
        progress = stats.get("current_file")
        if stats["bytes"] > last_bytes + 256 * 1024 or progress != last_progress:
            last_bytes = stats["bytes"]
            last_progress = progress
            stall_since = time.monotonic()
        elif time.monotonic() - stall_since > HF_DOWNLOAD_STALL_SEC:
            proc.terminate()
            proc.join(timeout=15)
            raise ArtifactError(
                f"Download stalled for {HF_DOWNLOAD_STALL_SEC}s at {stats['human']} "
                f"({progress or 'unknown file'}). Check network and disk at {stats['path']}"
            )

        proc.join(timeout=10)

    if proc.exitcode != 0:
        err_file = os.path.join(target, ".inferedge-download-error")
        detail = f"exit code {proc.exitcode}"
        if os.path.isfile(err_file):
            try:
                detail = open(err_file, encoding="utf-8").read().strip()
            except OSError:
                pass
        raise ArtifactError(f"Download failed for {model_id}: {detail}")


def _verify_hf_access(model_id: str) -> None:
    """Fail fast before a long snapshot_download when token/license is wrong."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if not HF_TOKEN and _is_likely_gated(model_id):
        raise _hf_auth_error(model_id)

    api = HfApi(token=HF_TOKEN or None)
    try:
        api.model_info(model_id)
    except (GatedRepoError, HfHubHTTPError) as exc:
        _raise_for_hf_http(model_id, exc)


def ensure_artifact(model_id: str, cache_dir: str = CACHE_DIR) -> str:
    model_key = normalize_model_key(model_id)
    target = os.path.join(cache_dir, model_key)
    if _cache_has_weights(target):
        return target

    if os.path.isdir(target):
        logger.warning("Removing incomplete model cache at %s", target)
        _clear_cache_dir(target)

    _verify_hf_access(model_id)
    _check_disk_space(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(
        "Downloading %s to %s (timeout=%ss, stall=%ss)",
        model_id,
        target,
        HF_DOWNLOAD_TIMEOUT,
        HF_DOWNLOAD_STALL_SEC,
    )
    try:
        _run_download_with_watchdog(model_id, target, cache_dir)

        stats = get_cache_stats(model_id, cache_dir)
        logger.info(
            "Download complete for %s: %s, %d weight file(s) at %s",
            model_id,
            stats["human"],
            stats["weight_files"],
            target,
        )
        if not _cache_has_weights(target):
            _clear_cache_dir(target)
            raise ArtifactError(
                f"Download finished but no model weights found for {model_id}. "
                f"Check HF_TOKEN and license access at https://huggingface.co/{model_id}"
            )
        return target
    except (GatedRepoError, HfHubHTTPError) as exc:
        _clear_cache_dir(target)
        _raise_for_hf_http(model_id, exc)
    except OSError as exc:
        if exc.errno == 28 or "No space left" in str(exc):
            raise ArtifactError(
                f"Disk full while downloading {model_id}. Free space under {cache_dir}."
            ) from exc
        raise ArtifactError(f"Filesystem error downloading {model_id}: {exc}") from exc
    except ArtifactError:
        _clear_cache_dir(target)
        raise
    except Exception as exc:
        _clear_cache_dir(target)
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg:
            raise _hf_auth_error(model_id) from exc
        if "403" in msg or "forbidden" in msg or "gated" in msg:
            raise (_hf_license_error(model_id) if HF_TOKEN else _hf_auth_error(model_id)) from exc
        if "connection" in msg or "timeout" in msg or "network" in msg:
            raise ArtifactError(f"Network error downloading {model_id}: {exc}") from exc
        raise ArtifactError(f"Failed to download {model_id}: {exc}") from exc


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


async def _apply_exit_records(records: list[dict]) -> None:
    for record in records:
        await state.update_deployment(**record)
        logger.error(
            "vLLM container %s exited unexpectedly (code=%s, generation=%s)",
            record["container_id"][:12],
            record["exit_code"],
            record.get("generation"),
        )


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
            container.stop(timeout=30)
            container.remove(force=True)
            stopped += 1
        except DockerException as exc:
            raise DockerError(f"Failed to stop container {container.id[:12]}: {exc}") from exc
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


def _start_or_update_vllm_sync(
    model_id: str,
    model_path: str,
    desired: DesiredState,
    config_hash: str,
    generation: int,
) -> tuple[str, Optional[dict]]:
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
            return container.id, None

    client = _get_client()
    env = {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {}
    command = [
        "vllm",
        "serve",
        model_path,
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
        container = client.containers.run(
            VLLM_IMAGE,
            command=command,
            detach=True,
            labels=labels,
            environment=env,
            volumes={CACHE_DIR: {"bind": CACHE_DIR, "mode": "rw"}},
            device_requests=device_requests,
            network=DOCKER_NETWORK,
            name=f"inferedge-vllm-gen{generation}",
            remove=False,
        )
        network = client.networks.get(DOCKER_NETWORK)
        try:
            network.disconnect(container.id, force=True)
        except DockerException:
            pass
        network.connect(container.id, aliases=[VLLM_ALIAS])
    except DockerException as exc:
        raise DockerError(f"Failed to start vLLM container: {exc}") from exc

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
        raise DockerError(
            f"vLLM container exited during startup (code={exit_record['exit_code']})"
        )

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


def _wait_for_probes_sync(model_id: str) -> ActualState:
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


async def wait_for_probes(model_id: str) -> ActualState:
    return await asyncio.to_thread(_wait_for_probes_sync, model_id)


def _get_deployment_status_sync(
    desired_model: Optional[str],
    record: dict,
) -> tuple[ActualState, list[dict]]:
    containers = find_managed_vllm_containers()
    running = [c for c in containers if c.status == "running"]
    exit_records: list[dict] = []

    for container in containers:
        if container.status not in ("running",):
            exit_code = container.attrs.get("State", {}).get("ExitCode")
            if exit_code not in (None, 0):
                exit_records.append(_exit_record_from_container(container, int(exit_code)))

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