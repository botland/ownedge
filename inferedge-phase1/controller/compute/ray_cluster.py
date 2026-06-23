"""External Ray cluster scheduler — active only in ray_cluster mode."""

import logging
import os

from compute.base import AbstractScheduler

logger = logging.getLogger(__name__)


class RayClusterScheduler(AbstractScheduler):
    """Connects to an external Ray cluster; no local ray.init(address='local')."""

    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        import ray

        address = os.environ.get("RAY_ADDRESS")
        if not address:
            raise ValueError("RAY_ADDRESS is required when COMPUTE_BACKEND=ray_cluster")
        ray.init(
            address=address,
            include_dashboard=False,
            logging_level=logging.WARNING,
            ignore_reinit_error=True,
        )
        self._started = True
        logger.info("RayClusterScheduler connected to %s", address)

    def shutdown(self) -> None:
        if self._started:
            import ray

            ray.shutdown()
            self._started = False
            logger.info("RayClusterScheduler disconnected")

    def is_ready(self) -> bool:
        if not self._started:
            return False
        import ray

        return ray.is_initialized()