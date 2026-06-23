"""Model artifact download and cache management.

Called by the reconciler only — not the API layer.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from exceptions import ArtifactError, TransientArtifactError
from serving.types import normalize_model_key

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("LOCAL_MODEL_CACHE", "/models_cache")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DOWNLOAD_TIMEOUT = int(os.environ.get("HF_DOWNLOAD_TIMEOUT_SEC", "7200"))
HF_DOWNLOAD_STALL_SEC = int(os.environ.get("HF_DOWNLOAD_STALL_SEC", "3600"))
HF_API_TIMEOUT = float(os.environ.get("HF_API_TIMEOUT_SEC", "60"))
MIN_FREE_DISK_GB = float(os.environ.get("MIN_FREE_DISK_GB", "20"))
DOWNLOAD_CONNECTIONS = int(os.environ.get("DOWNLOAD_CONNECTIONS", "16"))
_HF_INTERNAL_DIRS = frozenset({".hf_home", ".cache"})
GATED_MODEL_PREFIXES = ("meta-llama/", "google/gemma-")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_SHARD_RE = re.compile(r"^(.+)-(\d+)-of-(\d+)(\.[^.]+)$")
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


def ensure_model_metadata(model_id: str, target: str) -> None:
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
        ensure_model_metadata(model_id, target)
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
        ensure_model_metadata(model_id, target)
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

