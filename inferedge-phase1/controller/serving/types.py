"""Shared serving types and pure helpers."""

import hashlib
import json
import logging
import os

from schemas import DesiredState

logger = logging.getLogger(__name__)


def normalize_model_key(model_id: str) -> str:
    return model_id.replace("/", "--")


def compute_config_hash(desired: DesiredState) -> str:
    payload = json.dumps(
        {
            "model": desired.model,
            "context_length": desired.context_length,
            "gpu_utilization": desired.gpu_utilization,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def apply_gpu_profile(desired: DesiredState, model_id: str) -> DesiredState:
    """Apply GPU_PROFILE=auto context and utilization tuning before config_hash."""
    if os.environ.get("GPU_PROFILE", "auto") != "auto":
        return desired
    from gpu import total_vram_mb, tune_gpu_settings

    capped_ctx, tuned_util = tune_gpu_settings(
        model_id, desired.context_length, desired.gpu_utilization
    )
    updates: dict[str, float | int] = {}
    if capped_ctx < desired.context_length:
        updates["context_length"] = capped_ctx
    if tuned_util > desired.gpu_utilization:
        updates["gpu_utilization"] = tuned_util
    if not updates:
        return desired
    logger.info(
        "GPU_PROFILE=auto: tuned %s (%d MB VRAM): context %d -> %d, util %.2f -> %.2f",
        model_id,
        total_vram_mb(),
        desired.context_length,
        capped_ctx,
        desired.gpu_utilization,
        tuned_util,
    )
    return desired.model_copy(update=updates)
