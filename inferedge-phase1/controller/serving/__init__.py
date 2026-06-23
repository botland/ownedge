"""Serving backend factory and stack wiring."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compute.base import AbstractScheduler
    from serving.base import AbstractServingBackend


@dataclass
class ServingStack:
    mode: str
    backend: "AbstractServingBackend"
    scheduler: "AbstractScheduler | None"


def _normalize_mode(mode: str) -> str:
    if mode in ("local", "litellm_vllm", ""):
        return "litellm_vllm"
    return mode


def get_serving_stack() -> ServingStack:
    """Select serving backend and optional compute scheduler from COMPUTE_BACKEND."""
    mode = _normalize_mode(os.environ.get("COMPUTE_BACKEND", "litellm_vllm"))
    if mode == "litellm_vllm":
        from serving.docker_vllm import DockerVllmServingBackend

        return ServingStack(mode, DockerVllmServingBackend(), None)
    if mode == "ray_cluster":
        from compute.ray_cluster import RayClusterScheduler
        from serving.ray_cluster import RayClusterServingBackend

        scheduler = RayClusterScheduler()
        return ServingStack(mode, RayClusterServingBackend(scheduler), scheduler)
    raise ValueError(f"Unknown COMPUTE_BACKEND: {mode}")


def get_scheduler() -> "AbstractScheduler | None":
    """Backward-compatible helper; returns scheduler only in ray_cluster mode."""
    return get_serving_stack().scheduler