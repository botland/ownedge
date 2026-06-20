"""vLLM lifecycle via Docker labels.

IMPORTANT: This module is the sole Docker touchpoint and must only be called
from the reconciler. No API-layer imports.
"""

import asyncio
import hashlib
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
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

import state
from exceptions import (
    ArtifactError,
    DockerError,
    ProbeTimeoutError,
    TransientArtifactError,
    TransientDockerError,
)
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

_docker_client: docker.DockerClient | None = None
_vllm_op_lock = threading.Lock()


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
        for root, files in _walk_model_files(target):
            for name in files:
                if name == ".inferedge-download-progress":
                    continue
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                total_bytes += size
                if name.endswith(WEIGHT_SUFFIXES) and ".incomplete" not in name and not name.endswith(
                    ".part"
                ):
                    weight_files += 1
    return {
        "path": target,
        "bytes": total_bytes,
        "human": _format_bytes(total_bytes),
        "weight_files": weight_files,
        "current_file": current_file,
    }


_SHARD_RE = re.compile(r"^(.+)-(\d+)-of-(\d+)(\.[^.]+)$")


def _walk_model_files(target: str):
    """Walk model cache files, skipping legacy HF hub metadata dirs."""
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in _HF_INTERNAL_DIRS]
        yield root, files


def _cache_has_config(target: str) -> bool:
    return os.path.isfile(os.path.join(target, "config.json"))


_REQUIRED_JSON_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)
_JSON_SIZE_LIMITS: dict[str, int] = {
    "config.json": 1 * 1024 * 1024,
    "tokenizer_config.json": 1 * 1024 * 1024,
    "special_tokens_map.json": 1 * 1024 * 1024,
    "generation_config.json": 1 * 1024 * 1024,
    "model.safetensors.index.json": 50 * 1024 * 1024,
    "tokenizer.json": 200 * 1024 * 1024,
}


def _json_file_valid(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as json_f:
            json.load(json_f)
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False


def _metadata_needs_redownload(path: str, filename: str) -> bool:
    if not os.path.isfile(path):
        return True
    try:
        size = os.path.getsize(path)
    except OSError:
        return True
    limit = _JSON_SIZE_LIMITS.get(filename)
    if limit is not None and size > limit:
        return True
    return not _json_file_valid(path)


def _validate_model_cache(model_id: str, target: str) -> list[str]:
    """Return metadata filenames that are missing or corrupt."""
    missing: list[str] = []
    for name in _REQUIRED_JSON_FILES:
        path = os.path.join(target, name)
        if not _metadata_needs_redownload(path, name):
            continue
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
            logger.warning("Removed corrupt metadata %s for %s", name, model_id)
        missing.append(name)

    index_name = "model.safetensors.index.json"
    index_path = os.path.join(target, index_name)
    if os.path.isfile(index_path) and _metadata_needs_redownload(index_path, index_name):
        try:
            os.remove(index_path)
        except OSError:
            pass
        logger.warning("Removed corrupt metadata %s for %s", index_name, model_id)
        missing.append(index_name)
    return missing


def _repair_model_metadata(model_id: str, target: str, filenames: list[str]) -> None:
    os.makedirs(target, exist_ok=True)
    for filename in filenames:
        logger.info("Re-downloading metadata %s for %s", filename, model_id)
        _download_file(model_id, filename, target)


def _ensure_model_metadata(model_id: str, target: str) -> None:
    missing = _validate_model_cache(model_id, target)
    if not missing:
        return
    _repair_model_metadata(model_id, target, missing)
    still_missing = _validate_model_cache(model_id, target)
    if still_missing:
        raise ArtifactError(
            f"Model metadata still invalid for {model_id}: {', '.join(still_missing)}. "
            f"Delete {target} and retry."
        )


def _cache_has_weights(target: str) -> bool:
    """True only when all expected weight shards/files are present (not partial)."""
    if not os.path.isdir(target):
        return False

    shard_groups: dict[str, dict[str, int | set[int]]] = {}
    for root, files in _walk_model_files(target):
        for name in files:
            if name.endswith(".part") or ".incomplete" in name:
                continue
            match = _SHARD_RE.match(name)
            if not match:
                continue
            prefix, index_s, total_s, _suffix = match.groups()
            key = f"{prefix}-of-{total_s}"
            group = shard_groups.setdefault(key, {"total": int(total_s), "indices": set()})
            try:
                if os.path.getsize(os.path.join(root, name)) <= 0:
                    continue
            except OSError:
                continue
            group["indices"].add(int(index_s))

    if shard_groups:
        complete = [g for g in shard_groups.values() if len(g["indices"]) == g["total"]]
        if complete:
            return True
        logger.warning(
            "Incomplete shard set under %s: %s",
            target,
            {k: (len(v["indices"]), v["total"]) for k, v in shard_groups.items()},
        )
        return False

    for root, files in _walk_model_files(target):
        for name in files:
            if name.endswith(".part") or ".incomplete" in name:
                continue
            if name.endswith(WEIGHT_SUFFIXES):
                try:
                    if os.path.getsize(os.path.join(root, name)) > 0:
                        return True
                except OSError:
                    continue
    return False


def _clear_cache_dir(target: str) -> None:
    import shutil

    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)


def _raise_for_hf_http(model_id: str, exc: Exception) -> None:
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
    try:
        files = api.list_repo_files(model_id, timeout=HF_API_TIMEOUT)
    except TypeError:
        files = api.list_repo_files(model_id)
    return [f for f in files if _should_download_file(f)]


def _check_disk_space(cache_dir: str) -> None:
    import shutil

    usage = shutil.disk_usage(cache_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < MIN_FREE_DISK_GB:
        raise ArtifactError(
            f"Only {free_gb:.1f} GB free under {cache_dir}; "
            f"need at least {MIN_FREE_DISK_GB:.0f} GB for an 8B model download."
        )


def _configure_hf_download_env(target: str) -> None:
    """Force classic HTTP downloads — XET buffers silently and appears stuck in Docker."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    hf_home = os.path.join(target, ".hf_home")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")


def heal_download_environment(target: str) -> None:
    """Auto-heal wedged HF downloads: locks, XET staging, zero-byte incomplete files."""
    locks = _cleanup_hf_locks(target)
    xet_dir = os.path.join(target, ".hf_home", "xet")
    if os.path.isdir(xet_dir):
        import shutil

        shutil.rmtree(xet_dir, ignore_errors=True)
        logger.info("Auto-heal: cleared HF XET staging at %s", xet_dir)
    removed_incomplete = 0
    removed_corrupt = 0
    if os.path.isdir(target):
        for root, files in _walk_model_files(target):
            for name in files:
                path = os.path.join(root, name)
                if ".incomplete" in name:
                    try:
                        if os.path.getsize(path) == 0:
                            os.remove(path)
                            removed_incomplete += 1
                    except OSError:
                        pass
                    continue
                if name == ".inferedge-download-progress":
                    continue
                rel = os.path.relpath(path, target)
                if _remove_if_corrupt(path, rel):
                    removed_corrupt += 1
    if locks or removed_incomplete or removed_corrupt:
        logger.info(
            "Auto-heal: removed %d lock(s), %d empty incomplete file(s), "
            "%d corrupt file(s) under %s",
            locks,
            removed_incomplete,
            removed_corrupt,
            target,
        )


def _cleanup_hf_locks(target: str) -> int:
    """Remove stale HF download locks that block resume after a crashed download."""
    removed = 0
    for sub in (
        os.path.join(target, ".cache", "huggingface", "download"),
        os.path.join(target, ".hf_home", "hub"),
    ):
        if not os.path.isdir(sub):
            continue
        for root, _, files in os.walk(sub):
            for name in files:
                if not name.endswith(".lock"):
                    continue
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed += 1
                    logger.warning("Removed HF download lock: %s", path)
                except OSError:
                    pass
    return removed


def _snapshot_download_sizes(target: str) -> dict[str, int]:
    """Track in-progress bytes (.part and .incomplete files)."""
    sizes: dict[str, int] = {}
    if not os.path.isdir(target):
        return sizes
    for root, _, files in os.walk(target):
        for name in files:
            if (
                ".incomplete" not in name
                and not name.endswith(".part")
                and not name.endswith(".aria2")
            ):
                continue
            path = os.path.join(root, name)
            try:
                sizes[path] = os.path.getsize(path)
            except OSError:
                pass
    return sizes


def _download_sizes_grew(before: dict[str, int], after: dict[str, int]) -> bool:
    for path, size in after.items():
        if size > before.get(path, 0) + 64 * 1024:
            return True
    return len(after) > len(before)


def _write_download_progress(target: str, message: str) -> None:
    try:
        with open(os.path.join(target, ".inferedge-download-progress"), "w", encoding="utf-8") as f:
            f.write(message)
    except OSError:
        pass


def _is_metadata_file(filename: str) -> bool:
    lower = filename.lower()
    return filename.endswith((".json", ".txt", ".jinja", ".model")) or "tokenizer" in lower


def _file_looks_corrupt(filename: str, size: int) -> bool:
    """Detect files polluted by wrong cross-file resume (e.g. 575 MB tokenizer.json)."""
    if size <= 0:
        return True
    if filename.endswith(WEIGHT_SUFFIXES):
        return False
    limit = _JSON_SIZE_LIMITS.get(filename)
    if limit is not None:
        return size > limit
    if _is_metadata_file(filename):
        return size > 50 * 1024 * 1024
    return False


def _remove_if_corrupt(path: str, filename: str) -> bool:
    """Delete path when it looks like a mismatched partial resume. Returns True if removed."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if not _file_looks_corrupt(filename, size):
        return False
    try:
        os.remove(path)
        logger.warning(
            "Removed corrupt cached %s (%s) — will re-download",
            filename,
            _format_bytes(size),
        )
        return True
    except OSError:
        return False


def _bootstrap_part_from_legacy(target: str, filename: str, part: str) -> int:
    """Migrate partial bytes from a file-specific HF .incomplete cache into .part."""
    if os.path.isfile(part):
        size = os.path.getsize(part)
        if _file_looks_corrupt(filename, size):
            os.remove(part)
            return 0
        return size

    dest = os.path.join(target, filename)
    basename = os.path.basename(filename)
    dest_dir = os.path.dirname(dest) or target
    best_size = 0
    best_path: str | None = None
    for root, _, files in os.walk(target):
        for name in files:
            if ".incomplete" not in name:
                continue
            if not (name == f"{basename}.incomplete" or name.startswith(f"{basename}.")):
                continue
            if os.path.join(root, name) == part:
                continue
            path = os.path.join(root, name)
            if os.path.dirname(path) != dest_dir:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > best_size:
                best_size = size
                best_path = path
    if best_path and best_size > 0:
        if _file_looks_corrupt(filename, best_size):
            return 0
        import shutil

        shutil.copy2(best_path, part)
        logger.info(
            "Resumed from legacy HF partial cache (%s) for %s",
            _format_bytes(best_size),
            filename,
        )
        return best_size
    return 0


def _prepare_resume_file(dest: str, part: str, target: str, filename: str) -> None:
    """Merge .part / legacy incomplete into dest so aria2/HTTP can resume."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        _remove_if_corrupt(dest, filename)
        if os.path.isfile(dest):
            return
    if os.path.isfile(part):
        os.replace(part, dest)
        logger.info(
            "Continuing partial download %s (%s)",
            filename,
            _format_bytes(os.path.getsize(dest)),
        )
        return
    _bootstrap_part_from_legacy(target, filename, part)
    if os.path.isfile(part) and os.path.getsize(part) > 0:
        os.replace(part, dest)
        logger.info(
            "Continuing partial download %s (%s)",
            filename,
            _format_bytes(os.path.getsize(dest)),
        )


def _download_file_aria2(model_id: str, filename: str, target: str) -> None:
    """Multi-connection download for large shards (much faster than single HTTP stream)."""
    import subprocess

    dest = os.path.join(target, filename)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest) or target, exist_ok=True)

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        _remove_if_corrupt(dest, filename)
        if os.path.isfile(dest):
            return

    _prepare_resume_file(dest, part, target, filename)

    url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
    conns = str(DOWNLOAD_CONNECTIONS)
    cmd = [
        "aria2c",
        "-x",
        conns,
        "-s",
        conns,
        "-k",
        "4M",
        "--file-allocation=none",
        "--continue=true",
        "--allow-overwrite=true",
        f"--max-connection-per-server={conns}",
        "--summary-interval=30",
        "-d",
        os.path.dirname(dest) or target,
        "-o",
        os.path.basename(dest),
    ]
    if HF_TOKEN:
        cmd.append(f"--header=Authorization: Bearer {HF_TOKEN}")
    cmd.append(url)

    logger.info("Parallel download %s (%s connections)", filename, conns)
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientArtifactError(f"aria2 timed out on {filename}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "")[-500:]
        raise TransientArtifactError(f"aria2 failed on {filename}: {stderr}") from exc

    logger.info("Completed %s (%s)", filename, _format_bytes(os.path.getsize(dest)))


def _download_file(model_id: str, filename: str, target: str) -> None:
    if filename.endswith(".safetensors"):
        _download_file_aria2(model_id, filename, target)
    else:
        _download_file_streaming(model_id, filename, target)


def _download_file_streaming(model_id: str, filename: str, target: str) -> None:
    """Stream download to <target>/<filename>.part with HTTP Range resume."""
    import httpx

    dest = os.path.join(target, filename)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest) or target, exist_ok=True)

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        _remove_if_corrupt(dest, filename)
        if os.path.isfile(dest):
            return

    offset = _bootstrap_part_from_legacy(target, filename, part)
    url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
    headers: dict[str, str] = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    timeout = httpx.Timeout(60.0, read=900.0)
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            for attempt in range(2):
                req_headers = dict(headers)
                if offset > 0:
                    req_headers["Range"] = f"bytes={offset}-"
                    logger.info("Resuming %s from %s", filename, _format_bytes(offset))

                last_log = offset
                with client.stream("GET", url, headers=req_headers) as response:
                    if response.status_code == 416:
                        logger.warning(
                            "Range resume invalid for %s at %s; restarting download",
                            filename,
                            _format_bytes(offset),
                        )
                        for path in (part, dest):
                            if os.path.isfile(path):
                                os.remove(path)
                        offset = 0
                        continue
                    if response.status_code not in (200, 206):
                        raise TransientArtifactError(
                            f"HTTP {response.status_code} downloading {filename}"
                        )

                    mode = "wb"
                    downloaded = 0
                    if response.status_code == 206 and offset > 0:
                        mode = "ab"
                        downloaded = offset
                    else:
                        offset = 0

                    with open(part, mode) as out:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            out.write(chunk)
                            downloaded += len(chunk)
                            if downloaded - last_log >= 100 * 1024 * 1024:
                                logger.info(
                                    "Progress %s: %s", filename, _format_bytes(downloaded)
                                )
                                last_log = downloaded
                break
            else:
                raise TransientArtifactError(
                    f"HTTP 416 downloading {filename} after resume reset"
                )
    except httpx.HTTPError as exc:
        raise TransientArtifactError(f"Network error downloading {filename}") from exc

    os.replace(part, dest)
    logger.info("Completed %s (%s)", filename, _format_bytes(os.path.getsize(dest)))


def _download_all_files(model_id: str, target: str) -> None:
    """Download HF repo files one at a time (runs in spawn subprocess)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _configure_hf_download_env(target)
    heal_download_environment(target)

    files = _list_repo_files(model_id)
    if not files:
        raise ArtifactError(f"No downloadable vLLM files found in {model_id}")

    os.makedirs(target, exist_ok=True)
    total = len(files)
    for idx, filename in enumerate(files, 1):
        dest = os.path.join(target, filename)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            if _remove_if_corrupt(dest, filename):
                pass
            else:
                logger.info(
                    "Skipping existing %s (%s)", filename, _format_bytes(os.path.getsize(dest))
                )
                continue

        progress = f"[{idx}/{total}] {filename}"
        _write_download_progress(target, progress)
        logger.info("Downloading %s %s", model_id, progress)
        try:
            _download_file(model_id, filename, target)
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

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_download_all_files, args=(model_id, target), daemon=True)
    proc.start()

    last_bytes = 0
    last_progress = None
    last_partial = _snapshot_download_sizes(target)
    stall_since = time.monotonic()
    idle_deadline = time.monotonic() + HF_DOWNLOAD_TIMEOUT

    while proc.is_alive():
        stats = get_cache_stats(model_id, cache_dir)
        progress = stats.get("current_file")
        if progress == "complete":
            proc.join(timeout=30)
            if not proc.is_alive():
                break
            logger.warning("Download subprocess hung after marking complete; terminating")
            proc.terminate()
            proc.join(timeout=15)
            break
        partial = _snapshot_download_sizes(target)
        made_progress = (
            stats["bytes"] > last_bytes + 256 * 1024
            or progress != last_progress
            or _download_sizes_grew(last_partial, partial)
        )
        if made_progress:
            last_bytes = stats["bytes"]
            last_progress = progress
            last_partial = partial
            stall_since = time.monotonic()
            idle_deadline = time.monotonic() + HF_DOWNLOAD_TIMEOUT
        elif time.monotonic() > idle_deadline:
            proc.terminate()
            proc.join(timeout=15)
            heal_download_environment(target)
            raise TransientArtifactError(
                f"Download idle for {HF_DOWNLOAD_TIMEOUT}s at {stats['human']} "
                f"({progress or 'unknown file'}) for {model_id}; retrying"
            )
        elif time.monotonic() - stall_since > HF_DOWNLOAD_STALL_SEC:
            proc.terminate()
            proc.join(timeout=15)
            heal_download_environment(target)
            raise TransientArtifactError(
                f"Download stalled for {HF_DOWNLOAD_STALL_SEC}s at {stats['human']} "
                f"({progress or 'unknown file'}); auto-heal applied, retrying"
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
        stats = get_cache_stats(model_id, cache_dir)
        heal_download_environment(target)
        if stats["bytes"] > 10_000_000:
            raise TransientArtifactError(
                f"Download interrupted for {model_id} ({stats['human']} cached); retrying"
            ) from None
        raise ArtifactError(f"Download failed for {model_id}: {detail}")


def _verify_hf_access(model_id: str) -> None:
    """Fail fast before a long snapshot_download when token/license is wrong."""
    from huggingface_hub import HfApi

    if not HF_TOKEN and _is_likely_gated(model_id):
        raise _hf_auth_error(model_id)

    api = HfApi(token=HF_TOKEN or None)
    try:
        try:
            api.model_info(model_id, timeout=HF_API_TIMEOUT)
        except TypeError:
            api.model_info(model_id)
    except (GatedRepoError, HfHubHTTPError) as exc:
        _raise_for_hf_http(model_id, exc)


def _download_marked_complete(target: str) -> bool:
    progress_path = os.path.join(target, ".inferedge-download-progress")
    if not os.path.isfile(progress_path):
        return False
    try:
        with open(progress_path, encoding="utf-8") as progress_f:
            return progress_f.read().strip() == "complete"
    except OSError:
        return False


def ensure_artifact(model_id: str, cache_dir: str = CACHE_DIR) -> str:
    model_key = normalize_model_key(model_id)
    target = os.path.join(cache_dir, model_key)
    if _cache_has_weights(target):
        _ensure_model_metadata(model_id, target)
        logger.info("Model cache ready for %s at %s", model_id, target)
        return target

    if os.path.isdir(target):
        stats = get_cache_stats(model_id, cache_dir)
        if stats["bytes"] < 10_000_000:
            logger.warning("Removing tiny/broken cache at %s (%s)", target, stats["human"])
            _clear_cache_dir(target)
        else:
            logger.info("Resuming partial download at %s (%s)", target, stats["human"])
            heal_download_environment(target)
            if _download_marked_complete(target) and stats["weight_files"] > 0:
                raise ArtifactError(
                    f"Download marked complete for {model_id} but weight validation failed. "
                    f"Remove {target} or delete stale .hf_home metadata and retry."
                )

    _verify_hf_access(model_id)
    _check_disk_space(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(
        "Downloading %s to %s (idle_timeout=%ss, stall=%ss)",
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
        _ensure_model_metadata(model_id, target)
        return target
    except ArtifactError as exc:
        if isinstance(exc, TransientArtifactError):
            heal_download_environment(target)
            raise
        _clear_cache_dir(target)
        raise
    except (GatedRepoError, HfHubHTTPError) as exc:
        _clear_cache_dir(target)
        _raise_for_hf_http(model_id, exc)
    except OSError as exc:
        if exc.errno == 28 or "No space left" in str(exc):
            raise ArtifactError(
                f"Disk full while downloading {model_id}. Free space under {cache_dir}."
            ) from exc
        raise ArtifactError(f"Filesystem error downloading {model_id}: {exc}") from exc
    except Exception as exc:
        _clear_cache_dir(target)
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg:
            raise _hf_auth_error(model_id) from exc
        if "403" in msg or "forbidden" in msg or "gated" in msg:
            raise (_hf_license_error(model_id) if HF_TOKEN else _hf_auth_error(model_id)) from exc
        if "connection" in msg or "timeout" in msg or "network" in msg:
            heal_download_environment(target)
            raise TransientArtifactError(f"Network error downloading {model_id}; retrying") from exc
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
            if status == "running":
                container.stop(timeout=15)
            container.remove(force=True)
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
            if container.status == "running":
                container.stop(timeout=15)
            container.remove(force=True)
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
    logger.info("Starting vLLM for %s (generation=%d)", model_id, generation)
    gpu_ids = get_gpu_uuids()
    labels = _build_labels(model_key, config_hash, generation, gpu_ids)

    for container in find_managed_vllm_containers():
        cl = container.labels or {}
        if (
            cl.get(CONFIG_HASH_LABEL) == config_hash
            and int(cl.get(GENERATION_LABEL, "0")) == generation
            and container.status == "running"
        ):
            logger.info("Reusing running vLLM container %s", container.id[:12])
            return container.id, None

    client = _get_client()
    _ensure_vllm_image(client)
    _ensure_model_metadata(model_id, model_path)
    env = {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {}
    command = [
        model_path,
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

    device_requests = [
        DeviceRequest(count=-1, capabilities=[["gpu"]]),
    ]

    logger.info(
        "Creating vLLM container %s on network %s (shm=%s, cache=%s:%s)",
        f"inferedge-vllm-gen{generation}",
        DOCKER_NETWORK,
        _format_bytes(VLLM_SHM_SIZE),
        MODEL_CACHE_HOST,
        CACHE_DIR,
    )
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
            name=f"inferedge-vllm-gen{generation}",
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
            snippet = (record.get("log_snippet") or "").strip().splitlines()
            detail = snippet[-1] if snippet else f"exit code {exit_code}"
            raise DockerError(f"vLLM container exited during startup ({detail})")


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