import pytest
from pydantic import ValidationError

from schemas import (
    ActualState,
    ApplianceState,
    ApplianceStatus,
    DesiredState,
    LoadModelRequest,
    ReconcileMetrics,
)


def test_desired_state_defaults():
    d = DesiredState(model="org/model")
    assert d.context_length == 8192
    assert d.gpu_utilization == 0.85


def test_actual_state_defaults():
    a = ActualState()
    assert a.model_loaded is False
    assert a.health == "UNKNOWN"


def test_load_model_request_optional_fields():
    req = LoadModelRequest(model="llama-3.1-8b")
    dumped = req.model_dump(exclude_none=True)
    assert dumped == {"model": "llama-3.1-8b"}


def test_load_model_request_all_fields():
    req = LoadModelRequest(model="m", context_length=2048, gpu_utilization=0.9)
    assert req.context_length == 2048
    assert req.gpu_utilization == 0.9


def test_appliance_status_requires_core_fields():
    with pytest.raises(ValidationError):
        ApplianceStatus(appliance_id="x", state=ApplianceState.BOOT)  # type: ignore[call-arg]


def test_reconcile_metrics():
    m = ReconcileMetrics(duration_ms=12.5, intents_processed=2, vllm_restarts=1)
    assert m.model_dump()["vllm_restarts"] == 1