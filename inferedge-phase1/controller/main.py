"""InferEdge Controller API.

Layering Rules
--------------
- This module is the API layer. It MUST NOT import models.py or call Docker.
- Mutations append to intent_log via state.append_intent() only.
- desired_state, appliance_state, deployments, and reconcile_log are written
  exclusively by the reconciler.
- Protected endpoints require Authorization: Bearer <CONTROLLER_API_TOKEN>.
  /health and /status are always public.
"""

import logging
import os

# Must be set before huggingface_hub is imported anywhere in this process.
os.environ["HF_HUB_DISABLE_XET"] = "1"
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import state
from compute import get_scheduler
from reconciler import Reconciler
from schemas import ApplianceState, ApplianceStatus, LoadModelRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

APPLIANCE_ID = os.environ.get("APPLIANCE_ID", "inferedge-dev-001")
CONTROLLER_PORT = int(os.environ.get("CONTROLLER_PORT", "8080"))
API_TOKEN = os.environ.get("CONTROLLER_API_TOKEN", "")

security = HTTPBearer(auto_error=False)
reconciler: Reconciler | None = None
scheduler = get_scheduler()


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> None:
    if not API_TOKEN:
        return
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reconciler
    await state.migrate()
    await state.seed_defaults()
    scheduler.start()
    reconciler = Reconciler(APPLIANCE_ID)
    task = __import__("asyncio").create_task(reconciler.run_loop())
    logger.info("Controller started (appliance_id=%s)", APPLIANCE_ID)
    yield
    reconciler.stop()
    task.cancel()
    try:
        await task
    except __import__("asyncio").CancelledError:
        pass
    scheduler.shutdown()
    await state.close_db()
    logger.info("Controller shut down")


app = FastAPI(title="InferEdge Controller", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "scheduler_ready": scheduler.is_ready()}


@app.get("/status", response_model=ApplianceStatus)
async def get_status():
    actual = await state.get_cached_actual()
    data = await state.build_status(APPLIANCE_ID, actual)
    return ApplianceStatus(
        appliance_id=data["appliance_id"],
        state=data["state"],
        desired=data["desired"],
        actual=data["actual"],
        last_reconcile_ts=data["last_reconcile_ts"],
        last_error=data["last_error"],
    )


@app.post("/models/load")
async def load_model(request: LoadModelRequest, _: None = Depends(verify_token)):
    payload = request.model_dump(exclude_none=True)
    sequence_id = await state.append_intent("load_model", payload)
    return {"accepted": True, "sequence_id": sequence_id, "message": "Intent queued for reconciliation"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=CONTROLLER_PORT)