import json
import os

import pytest

from schemas import ActualState, ApplianceState, DesiredState


@pytest.mark.asyncio
async def test_migrate_creates_schema(fresh_state):
    db = await fresh_state.get_db()
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ) as cur:
        tables = {row[0] for row in await cur.fetchall()}
    assert {
        "appliance_state",
        "desired_state",
        "deployments",
        "intent_log",
        "model_aliases",
        "reconcile_log",
        "schema_meta",
    }.issubset(tables)


@pytest.mark.asyncio
async def test_seed_defaults_inserts_desired_state(fresh_state, env_defaults, monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    await fresh_state.seed_defaults()
    desired = await fresh_state.get_desired_state()
    assert desired.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert desired.context_length == 8192


@pytest.mark.asyncio
async def test_seed_defaults_queues_intent_when_env_changes(fresh_state, monkeypatch):
    await fresh_state.seed_defaults()
    await fresh_state.update_desired_state(
        DesiredState(model="other/model", context_length=2048, gpu_utilization=0.9)
    )
    monkeypatch.setenv("DEFAULT_MODEL", "new/model")
    monkeypatch.setenv("DEFAULT_CONTEXT", "4096")
    await fresh_state.seed_defaults()

    db = await fresh_state.get_db()
    async with db.execute(
        "SELECT action, payload_json, processed FROM intent_log ORDER BY sequence_id DESC LIMIT 1"
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == "load_model"
    assert json.loads(row[1])["model"] == "new/model"
    assert row[2] == 0


@pytest.mark.asyncio
async def test_resolve_model_alias(fresh_state):
    await fresh_state.seed_defaults()
    assert await fresh_state.resolve_model("default") == "meta-llama/Llama-3.1-8B-Instruct"
    assert await fresh_state.resolve_model("llama-3.1-8b") == "meta-llama/Llama-3.1-8B-Instruct"
    assert await fresh_state.resolve_model("custom/org-model") == "custom/org-model"


@pytest.mark.asyncio
async def test_append_and_fold_intents_last_wins(fresh_state):
    await fresh_state.seed_defaults()
    await fresh_state.append_intent("load_model", {"model": "first/model", "context_length": 1024})
    await fresh_state.append_intent("load_model", {"model": "second/model", "context_length": 2048})

    processed = await fresh_state.fold_intents_into_desired()
    assert processed == 2
    desired = await fresh_state.get_desired_state()
    assert desired.model == "second/model"
    assert desired.context_length == 2048

    db = await fresh_state.get_db()
    async with db.execute("SELECT processed FROM intent_log") as cur:
        rows = await cur.fetchall()
    assert all(row[0] == 1 for row in rows)


@pytest.mark.asyncio
async def test_appliance_state_roundtrip(fresh_state):
    actual = ActualState(health="HEALTHY", model_loaded=True)
    await fresh_state.set_appliance_state(
        ApplianceState.READY,
        last_error=None,
        last_reconcile_ts=12345.0,
        actual=actual,
    )
    state, err, ts = await fresh_state.get_appliance_state()
    assert state == ApplianceState.READY
    assert err is None
    assert ts == 12345.0

    cached = await fresh_state.get_cached_actual()
    assert cached.health == "HEALTHY"
    assert cached.model_loaded is True


@pytest.mark.asyncio
async def test_deployment_record_update(fresh_state):
    await fresh_state.update_deployment(
        container_id="cid-1",
        config_hash="hash-abc",
        generation=2,
        gpu_ids="GPU-1",
        model_key="org--model",
        exit_code=None,
        log_snippet=None,
    )
    record = await fresh_state.get_deployment_record()
    assert record["container_id"] == "cid-1"
    assert record["generation"] == 2


@pytest.mark.asyncio
async def test_get_next_generation(fresh_state):
    await fresh_state.update_deployment(generation=4)
    assert await fresh_state.get_next_generation() == 5


@pytest.mark.asyncio
async def test_log_reconcile_event(fresh_state):
    await fresh_state.log_reconcile_event("test_event", {"foo": "bar"})
    db = await fresh_state.get_db()
    async with db.execute("SELECT event, metrics_json FROM reconcile_log") as cur:
        row = await cur.fetchone()
    assert row[0] == "test_event"
    assert json.loads(row[1])["foo"] == "bar"


@pytest.mark.asyncio
async def test_build_status(fresh_state, sample_desired):
    await fresh_state.seed_defaults()
    await fresh_state.set_appliance_state(ApplianceState.RECONCILING, last_reconcile_ts=99.0)
    actual = ActualState(health="DOWNLOADING", download_bytes=1024)
    status = await fresh_state.build_status("test-appliance", actual)
    assert status["appliance_id"] == "test-appliance"
    assert status["state"] == ApplianceState.RECONCILING
    assert status["last_reconcile_ts"] == 99.0
    assert status["desired"].model == sample_desired.model