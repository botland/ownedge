import os

from compute.base import AbstractScheduler
from compute.local import LocalScheduler


def get_scheduler() -> AbstractScheduler:
    backend = os.environ.get("COMPUTE_BACKEND", "local")
    if backend == "local":
        return LocalScheduler()
    raise ValueError(f"Unknown compute backend: {backend}")