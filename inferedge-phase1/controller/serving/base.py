"""Abstract serving backend interface."""

from abc import ABC, abstractmethod

from schemas import ActualState, DesiredState


class AbstractServingBackend(ABC):
    @property
    @abstractmethod
    def mode(self) -> str:
        """litellm_vllm or ray_cluster."""

    @abstractmethod
    async def prewarm(self) -> None:
        """Pre-pull images or warm remote runtime."""

    @abstractmethod
    async def get_deployment_status(self, desired_model: str | None) -> ActualState:
        ...

    @abstractmethod
    async def stop_if_needed(
        self, *, except_hash: str | None, except_generation: int | None
    ) -> int:
        ...

    @abstractmethod
    async def start_or_update(
        self,
        model_id: str,
        model_path: str,
        desired: DesiredState,
        config_hash: str,
        generation: int,
    ) -> str:
        ...

    @abstractmethod
    async def wait_for_probes(self, model_id: str) -> ActualState:
        ...

    async def get_start_progress(self) -> dict:
        return {}

    async def get_load_hint(self, deployment_id: str | None) -> str | None:
        return None

    async def is_running(self) -> bool:
        return False

    async def heal_environment(self, force_restart_running: bool = False) -> list[str]:
        return []

    async def prune_exited(self) -> list[str]:
        return []

    def has_load_failure(self, config_hash: str, deployment: dict) -> bool:
        return (
            deployment.get("config_hash") == config_hash
            and deployment.get("exit_code") not in (None, 0)
        )

    def format_load_error(self, record: dict) -> str:
        from serving.load_errors import format_vllm_load_error

        return format_vllm_load_error(record)
