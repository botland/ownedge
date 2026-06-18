import logging
import subprocess
from dataclasses import dataclass, field
from typing import List

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
        import pynvml

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