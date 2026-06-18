import logging
import os

import ray

from compute.base import AbstractScheduler

logger = logging.getLogger(__name__)


class LocalScheduler(AbstractScheduler):
    """Single-node Ray scheduler running in-process with the controller."""

    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        dashboard_port = int(os.environ.get("RAY_DASHBOARD_PORT", "8265"))
        ray.init(
            address="local",
            include_dashboard=True,
            dashboard_port=dashboard_port,
            logging_level=logging.WARNING,
            ignore_reinit_error=True,
        )
        self._started = True
        logger.info("LocalScheduler started (Ray single-node, dashboard=%s)", dashboard_port)

    def shutdown(self) -> None:
        if self._started:
            ray.shutdown()
            self._started = False
            logger.info("LocalScheduler shut down")

    def is_ready(self) -> bool:
        return self._started and ray.is_initialized()