from abc import ABC, abstractmethod


class AbstractScheduler(ABC):
    """Abstract compute scheduler adapter.

    Phase 1 uses LocalScheduler (in-process Ray). Future phases may swap in
    a multi-node Ray implementation without changing the reconciler or API.
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