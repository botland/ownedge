import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GpuDevice:
    index: int
    uuid: str
    name: str
    total_vram_mb: int
    free_vram_mb: int


@dataclass
class GpuInfo:
    available: bool
    devices: List[GpuDevice] = field(default_factory=list)
    error: str | None = None


def _via_pynvml() -> GpuInfo:
    try:
        import pynvml  # provided by nvidia-ml-py

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        devices: list[GpuDevice] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                GpuDevice(
                    index=i,
                    uuid=uuid,
                    name=name,
                    total_vram_mb=mem.total // (1024 * 1024),
                    free_vram_mb=mem.free // (1024 * 1024),
                )
            )
        pynvml.nvmlShutdown()
        return GpuInfo(available=len(devices) > 0, devices=devices)
    except Exception as exc:
        logger.debug("pynvml unavailable: %s", exc)
        return GpuInfo(available=False, error=str(exc))


def _via_nvidia_smi() -> GpuInfo:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return GpuInfo(available=False, error=result.stderr.strip() or "nvidia-smi failed")

        devices: list[GpuDevice] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            devices.append(
                GpuDevice(
                    index=int(parts[0]),
                    uuid=parts[1],
                    name=parts[2],
                    total_vram_mb=int(float(parts[3])),
                    free_vram_mb=int(float(parts[4])),
                )
            )
        return GpuInfo(available=len(devices) > 0, devices=devices)
    except Exception as exc:
        return GpuInfo(available=False, error=str(exc))


def get_gpu_info() -> GpuInfo:
    info = _via_pynvml()
    if info.available:
        return info
    return _via_nvidia_smi()


def is_gpu_available() -> bool:
    return get_gpu_info().available


def get_gpu_uuids() -> list[str]:
    return [d.uuid for d in get_gpu_info().devices]


def total_vram_mb() -> int:
    return sum(d.total_vram_mb for d in get_gpu_info().devices)


_CONTEXT_STEPS = (8192, 4096, 3072, 2048, 1536, 1024, 512)


def _is_quantized(model_id: str) -> bool:
    model_lower = model_id.lower()
    return any(token in model_lower for token in ("awq", "gptq", "gguf", "int4", "int8", "bnb"))


def _param_billions(model_id: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
    if not match:
        return None
    return float(match.group(1))


def _estimate_weight_gb(model_id: str) -> Optional[float]:
    """Rough fp16 weight footprint from model id (e.g. Llama-3.1-8B -> ~16 GB)."""
    param_b = _param_billions(model_id)
    if param_b is None:
        return None
    multiplier = 0.67 if _is_quantized(model_id) else 2.0
    return param_b * multiplier


def _kv_cache_gb(model_id: str, context_length: int) -> float:
    """Estimate KV-cache VRAM from model size and context (empirical for GQA Llama-family)."""
    param_b = _param_billions(model_id) or 8.0
    return param_b * 0.00005 * context_length


def _activation_overhead_gb(model_id: str) -> float:
    return 1.2 if _is_quantized(model_id) else 2.0


def _vram_fits(model_id: str, context_length: int, gpu_utilization: float) -> bool:
    info = get_gpu_info()
    if not info.available:
        return True
    weight_gb = _estimate_weight_gb(model_id)
    if weight_gb is None:
        return True
    total_gb = sum(d.total_vram_mb for d in info.devices) / 1024.0
    kv_budget_gb = (
        total_gb * gpu_utilization
        - weight_gb
        - _activation_overhead_gb(model_id)
    )
    return kv_budget_gb >= _kv_cache_gb(model_id, context_length) * 1.05


def check_vram_for_model(
    model_id: str, context_length: int, gpu_utilization: float = 0.85
) -> Optional[str]:
    """Return an operator-facing message when GPU VRAM is likely too small."""
    info = get_gpu_info()
    if not info.available:
        return None

    weight_gb = _estimate_weight_gb(model_id)
    if weight_gb is None:
        return None

    if _vram_fits(model_id, context_length, gpu_utilization):
        return None

    total_gb = sum(d.total_vram_mb for d in info.devices) / 1024.0
    kv_needed_gb = _kv_cache_gb(model_id, context_length)
    kv_budget_gb = (
        total_gb * gpu_utilization
        - weight_gb
        - _activation_overhead_gb(model_id)
    )
    capped = cap_context_length(model_id, context_length, gpu_utilization)
    quant_hint = ""
    if not _is_quantized(model_id):
        quant_hint = " Use a quantized variant (AWQ/GPTQ) or a smaller model."
    context_hint = ""
    if capped < context_length:
        context_hint = f" Try context_length={capped} or lower on this GPU."
    return (
        f"GPU VRAM likely insufficient for {model_id} at context {context_length}: "
        f"~{kv_needed_gb:.1f} GB KV cache needed, ~{max(kv_budget_gb, 0):.1f} GB KV budget "
        f"on {info.devices[0].name}.{context_hint} "
        f"Try gpu_utilization>=0.95 or lower context_length.{quant_hint}"
    )


def tune_gpu_settings(
    model_id: str, context_length: int, gpu_utilization: float
) -> tuple[int, float]:
    """Pick context/gpu_util that fits when GPU_PROFILE=auto."""
    util_steps = sorted({gpu_utilization, 0.95, 0.98})
    best_ctx = cap_context_length(model_id, context_length, gpu_utilization)
    best_util = gpu_utilization
    for util in util_steps:
        capped = cap_context_length(model_id, context_length, util)
        if capped > best_ctx or (
            capped == best_ctx and util > best_util
        ):
            best_ctx = capped
            best_util = util
    return best_ctx, best_util


def cap_context_length(
    model_id: str, requested: int, gpu_utilization: float = 0.85
) -> int:
    """Highest supported context length that fits the current GPU budget."""
    candidates = [step for step in _CONTEXT_STEPS if step <= requested]
    for context_length in sorted(candidates, reverse=True):
        if _vram_fits(model_id, context_length, gpu_utilization):
            return context_length
    return 512