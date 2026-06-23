from abc import ABC, abstractmethod


class AbstractScheduler(ABC):
    """Abstract compute scheduler adapter.

    Active only in ray_cluster mode (external Ray). litellm_vllm mode has no scheduler.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize the compute backend."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down the compute backend."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the scheduler is operational."""