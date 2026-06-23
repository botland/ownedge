"""Compute scheduler — only active in ray_cluster mode."""

from compute.base import AbstractScheduler


def get_scheduler() -> AbstractScheduler | None:
    from serving import get_serving_stack

    return get_serving_stack().scheduler


__all__ = ["AbstractScheduler", "get_scheduler"]