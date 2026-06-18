from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ApplianceState(str, Enum):
    BOOT = "BOOT"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    RECONCILING = "RECONCILING"


class DesiredState(BaseModel):
    model: str
    context_length: int = 8192
    gpu_utilization: float = 0.85


class ActualState(BaseModel):
    model_loaded: bool = False
    current_model: Optional[str] = None
    container_id: Optional[str] = None
    health: str = "UNKNOWN"
    config_hash: Optional[str] = None
    generation: Optional[int] = None
    gpu_ids: Optional[str] = None
    exit_code: Optional[int] = None
    log_snippet: Optional[str] = None
    download_bytes: Optional[int] = None
    download_weight_files: Optional[int] = None
    download_current_file: Optional[str] = None


class ApplianceStatus(BaseModel):
    appliance_id: str
    state: ApplianceState
    desired: DesiredState
    actual: ActualState
    last_reconcile_ts: Optional[float] = None
    last_error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class LoadModelRequest(BaseModel):
    model: str
    context_length: Optional[int] = None
    gpu_utilization: Optional[float] = None


class IntentRecord(BaseModel):
    sequence_id: int
    action: str
    payload_json: str
    processed: bool = False
    created_at: str


class ReconcileMetrics(BaseModel):
    duration_ms: float
    intents_processed: int = 0
    vllm_restarts: int = 0