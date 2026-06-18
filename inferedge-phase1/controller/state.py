"""SQLite state layer.

Single-writer conventions:
- API layer may call append_intent() only.
- Reconciler may call update_desired_state(), set_appliance_state(),
  update_deployment(), log_reconcile_event(), and fold_intents_into_desired().
"""

import json
import logging
import os
import time
from typing import Any, Optional

import aiosqlite

from schemas import ActualState, ApplianceState, DesiredState

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DB_PATH = os.environ.get("SQLITE_DB_PATH", "/data/inferedge.db")

_db: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    global _db
    if _db is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _db = await aiosqlite.connect(DB_PATH)
        _db.row_factory = aiosqlite.Row
    return _db


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None


async def _get_schema_version(db: aiosqlite.Connection) -> int:
    try:
        async with db.execute("SELECT MAX(version) AS v FROM schema_meta") as cur:
            row = await cur.fetchone()
            return int(row["v"]) if row and row["v"] is not None else 0
    except aiosqlite.OperationalError:
        return 0


async def _migration_v1(db: aiosqlite.Connection) -> None:
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS appliance_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            state TEXT NOT NULL DEFAULT 'BOOT',
            last_error TEXT,
            last_reconcile_ts REAL,
            actual_json TEXT
        );

        CREATE TABLE IF NOT EXISTS desired_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model TEXT NOT NULL,
            context_length INTEGER NOT NULL DEFAULT 8192,
            gpu_utilization REAL NOT NULL DEFAULT 0.85
        );

        CREATE TABLE IF NOT EXISTS intent_log (
            sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_aliases (
            alias TEXT PRIMARY KEY,
            model_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            container_id TEXT,
            config_hash TEXT,
            generation INTEGER NOT NULL DEFAULT 0,
            gpu_ids TEXT,
            model_key TEXT,
            exit_code INTEGER,
            log_snippet TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS reconcile_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            metrics_json TEXT,
            created_at TEXT NOT NULL
        );

        INSERT OR IGNORE INTO appliance_state (id, state) VALUES (1, 'BOOT');
        INSERT OR IGNORE INTO deployments (id, generation) VALUES (1, 0);
        """
    )


MIGRATIONS: dict[int, Any] = {
    1: _migration_v1,
}


async def migrate() -> None:
    """Apply pending schema migrations.

    To add a future migration: increment SCHEMA_VERSION and register a new
    function in MIGRATIONS (e.g. 2: _migration_v2).
    """
    db = await get_db()
    current = await _get_schema_version(db)
    for version in range(current + 1, SCHEMA_VERSION + 1):
        fn = MIGRATIONS.get(version)
        if fn is None:
            raise RuntimeError(f"No migration defined for schema version {version}")
        logger.info("Applying schema migration v%s", version)
        await fn(db)
        await db.execute(
            "INSERT INTO schema_meta (version, applied_at) VALUES (?, ?)",
            (version, _now_iso()),
        )
        await db.commit()
    logger.info("Schema at version %s", await _get_schema_version(db))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def seed_defaults() -> None:
    """Seed SQLite and sync .env model settings into the intent log when they change."""
    db = await get_db()
    default_model = os.environ.get("DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    default_context = int(os.environ.get("DEFAULT_CONTEXT", "8192"))
    default_util = float(os.environ.get("GPU_UTILIZATION", "0.85"))

    await db.execute(
        """INSERT INTO model_aliases (alias, model_id) VALUES (?, ?)
           ON CONFLICT(alias) DO UPDATE SET model_id = excluded.model_id""",
        ("default", default_model),
    )
    await db.execute(
        "INSERT OR IGNORE INTO model_aliases (alias, model_id) VALUES (?, ?)",
        ("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    await db.commit()

    resolved_env_model = await resolve_model(default_model)

    async with db.execute("SELECT model FROM desired_state WHERE id = 1") as cur:
        row = await cur.fetchone()
    if row is None:
        await db.execute(
            "INSERT INTO desired_state (id, model, context_length, gpu_utilization) VALUES (1, ?, ?, ?)",
            (resolved_env_model, default_context, default_util),
        )
        await db.commit()
        logger.info("Seeded desired state from env: model=%s", resolved_env_model)
        return

    stored = await get_desired_state()
    env_changed = (
        stored.model != resolved_env_model
        or stored.context_length != default_context
        or abs(stored.gpu_utilization - default_util) > 1e-6
    )
    if env_changed:
        sequence_id = await append_intent(
            "load_model",
            {
                "model": default_model,
                "context_length": default_context,
                "gpu_utilization": default_util,
            },
        )
        logger.info(
            "Queued model change from .env (sequence_id=%s): %s -> %s",
            sequence_id,
            stored.model,
            resolved_env_model,
        )


async def resolve_model(model_or_alias: str) -> str:
    db = await get_db()
    async with db.execute(
        "SELECT model_id FROM model_aliases WHERE alias = ?", (model_or_alias,)
    ) as cur:
        row = await cur.fetchone()
    if row:
        return row["model_id"]
    return model_or_alias


async def append_intent(action: str, payload: dict) -> int:
    """API layer only — append a mutation intent to the log."""
    db = await get_db()
    await db.execute(
        "INSERT INTO intent_log (action, payload_json, created_at) VALUES (?, ?, ?)",
        (action, json.dumps(payload), _now_iso()),
    )
    await db.commit()
    async with db.execute("SELECT last_insert_rowid() AS id") as cur:
        row = await cur.fetchone()
    return int(row["id"])


async def fold_intents_into_desired() -> int:
    """Reconciler only — apply unprocessed intents to desired_state."""
    db = await get_db()
    processed = 0
    async with db.execute(
        "SELECT sequence_id, action, payload_json FROM intent_log WHERE processed = 0 ORDER BY sequence_id"
    ) as cur:
        rows = await cur.fetchall()

    for row in rows:
        payload = json.loads(row["payload_json"])
        if row["action"] == "load_model":
            model = await resolve_model(payload["model"])
            context = payload.get("context_length")
            util = payload.get("gpu_utilization")
            async with db.execute("SELECT * FROM desired_state WHERE id = 1") as dcur:
                desired = await dcur.fetchone()
            if desired:
                context = context if context is not None else desired["context_length"]
                util = util if util is not None else desired["gpu_utilization"]
            else:
                context = context or int(os.environ.get("DEFAULT_CONTEXT", "8192"))
                util = util or float(os.environ.get("GPU_UTILIZATION", "0.85"))
            await db.execute(
                """INSERT INTO desired_state (id, model, context_length, gpu_utilization)
                   VALUES (1, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     model = excluded.model,
                     context_length = excluded.context_length,
                     gpu_utilization = excluded.gpu_utilization""",
                (model, context, util),
            )
        await db.execute(
            "UPDATE intent_log SET processed = 1 WHERE sequence_id = ?",
            (row["sequence_id"],),
        )
        processed += 1
    await db.commit()
    return processed


async def get_desired_state() -> DesiredState:
    db = await get_db()
    async with db.execute("SELECT * FROM desired_state WHERE id = 1") as cur:
        row = await cur.fetchone()
    if row is None:
        return DesiredState(
            model=os.environ.get("DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        )
    return DesiredState(
        model=row["model"],
        context_length=row["context_length"],
        gpu_utilization=row["gpu_utilization"],
    )


async def update_desired_state(desired: DesiredState) -> None:
    """Reconciler only — persist authoritative desired state."""
    db = await get_db()
    await db.execute(
        """INSERT INTO desired_state (id, model, context_length, gpu_utilization)
           VALUES (1, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             model = excluded.model,
             context_length = excluded.context_length,
             gpu_utilization = excluded.gpu_utilization""",
        (desired.model, desired.context_length, desired.gpu_utilization),
    )
    await db.commit()


async def get_appliance_state() -> tuple[ApplianceState, Optional[str], Optional[float]]:
    db = await get_db()
    async with db.execute("SELECT state, last_error, last_reconcile_ts FROM appliance_state WHERE id = 1") as cur:
        row = await cur.fetchone()
    if row is None:
        return ApplianceState.BOOT, None, None
    return ApplianceState(row["state"]), row["last_error"], row["last_reconcile_ts"]


async def set_appliance_state(
    state: ApplianceState,
    last_error: Optional[str] = None,
    last_reconcile_ts: Optional[float] = None,
    actual: Optional[ActualState] = None,
) -> None:
    """Reconciler only."""
    db = await get_db()
    actual_json = json.dumps(actual.model_dump()) if actual else None
    if actual_json is not None:
        await db.execute(
            """UPDATE appliance_state SET state = ?, last_error = ?, last_reconcile_ts = ?, actual_json = ?
               WHERE id = 1""",
            (state.value, last_error, last_reconcile_ts, actual_json),
        )
    else:
        await db.execute(
            """UPDATE appliance_state SET state = ?, last_error = ?, last_reconcile_ts = ?
               WHERE id = 1""",
            (state.value, last_error, last_reconcile_ts),
        )
    await db.commit()


async def get_cached_actual() -> ActualState:
    """Read last reconciled actual state (API-safe, no Docker)."""
    db = await get_db()
    async with db.execute("SELECT actual_json FROM appliance_state WHERE id = 1") as cur:
        row = await cur.fetchone()
    if row and row["actual_json"]:
        return ActualState(**json.loads(row["actual_json"]))
    record = await get_deployment_record()
    return ActualState(
        config_hash=record.get("config_hash"),
        generation=record.get("generation"),
        gpu_ids=record.get("gpu_ids"),
        exit_code=record.get("exit_code"),
        log_snippet=record.get("log_snippet"),
        health="UNKNOWN",
    )


async def get_deployment_record() -> dict:
    db = await get_db()
    async with db.execute("SELECT * FROM deployments WHERE id = 1") as cur:
        row = await cur.fetchone()
    if row is None:
        return {"generation": 0}
    return dict(row)


async def get_next_generation() -> int:
    record = await get_deployment_record()
    return int(record.get("generation") or 0) + 1


async def update_deployment(
    *,
    container_id: Optional[str] = None,
    config_hash: Optional[str] = None,
    generation: Optional[int] = None,
    gpu_ids: Optional[str] = None,
    model_key: Optional[str] = None,
    exit_code: Optional[int] = None,
    log_snippet: Optional[str] = None,
) -> None:
    """Reconciler only."""
    db = await get_db()
    record = await get_deployment_record()
    await db.execute(
        """UPDATE deployments SET
             container_id = ?,
             config_hash = ?,
             generation = ?,
             gpu_ids = ?,
             model_key = ?,
             exit_code = ?,
             log_snippet = ?,
             updated_at = ?
           WHERE id = 1""",
        (
            container_id if container_id is not None else record.get("container_id"),
            config_hash if config_hash is not None else record.get("config_hash"),
            generation if generation is not None else record.get("generation", 0),
            gpu_ids if gpu_ids is not None else record.get("gpu_ids"),
            model_key if model_key is not None else record.get("model_key"),
            exit_code if exit_code is not None else record.get("exit_code"),
            log_snippet if log_snippet is not None else record.get("log_snippet"),
            _now_iso(),
        ),
    )
    await db.commit()


async def log_reconcile_event(event: str, metrics: Optional[dict] = None) -> None:
    """Reconciler only — append on real changes only."""
    db = await get_db()
    await db.execute(
        "INSERT INTO reconcile_log (event, metrics_json, created_at) VALUES (?, ?, ?)",
        (event, json.dumps(metrics) if metrics else None, _now_iso()),
    )
    await db.commit()


async def build_status(appliance_id: str, actual: ActualState) -> dict:
    state, last_error, last_reconcile_ts = await get_appliance_state()
    desired = await get_desired_state()
    resolved_model = await resolve_model(desired.model)
    desired_display = desired.model_copy(update={"model": resolved_model})
    return {
        "appliance_id": appliance_id,
        "state": state,
        "desired": desired_display,
        "actual": actual,
        "last_reconcile_ts": last_reconcile_ts,
        "last_error": last_error,
    }