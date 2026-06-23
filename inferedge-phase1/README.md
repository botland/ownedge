# InferEdge Phase 1

Single-node inference appliance with a Controller reconciler, label-managed vLLM, LiteLLM gateway, and Traefik routing.

## Prerequisites

- Ubuntu 24.04 (or similar Linux)
- NVIDIA GPU with ≥8 GB VRAM (≥16 GB recommended)
- Docker Compose v2
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Hugging Face token (`HF_TOKEN`) — **required** for gated models like `meta-llama/Llama-3.1-8B-Instruct`

## Quick Start

```bash
cp .env.example .env
# Edit .env: set CONTROLLER_API_TOKEN and HF_TOKEN (required for Llama)
# Accept the model license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

./scripts/compose.sh up -d --build
./scripts/compose.sh logs -f controller
```

## Check Status

```bash
curl http://localhost:8080/status | jq
```

`/status` is available immediately on boot (state `BOOT` → `RECONCILING` → `READY`).

## Auto-Healing Behavior

The reconciler retries automatically every `RECONCILE_INTERVAL_SEC` (default 5s). You should **not** need to manually clear locks or cache for transient issues.

| Failure type | State | What happens |
|---|---|---|
| Download stall / timeout / network blip | `RECONCILING` | Clears HF locks + XET staging, resumes partial cache, retries |
| HF auth / license (401/403) | `DEGRADED` | Stops — fix `HF_TOKEN` or accept model license |
| Disk full | `DEGRADED` | Stops — free space under `MODEL_CACHE_HOST` |
| vLLM crash during model load (CUDA/driver, container exit) | `DEGRADED` | Stops retrying for same config — fix driver/image, then change model/config or restart controller |
| vLLM Docker daemon error (create/stop) | `FAILED` → retry | Next cycle retries container operations |
| No GPU | `DEGRADED` | Expected CPU-only mode |

Watch auto-retry in logs: `Transient download issue (will retry)` or `/status` with `last_error: "Auto-retry: ..."`.

## Changing the Default Model

Desired state is stored in SQLite and survives reboots. To switch models:

**Option A — edit `.env` (recommended for appliance config)**

```bash
# Edit DEFAULT_MODEL in .env, then recreate the controller to pick up env vars:
./scripts/compose.sh up -d --force-recreate controller
```

On startup the controller compares `.env` to SQLite and queues a `load_model` intent if they differ.

**Option B — runtime API**

Protected endpoint — requires Bearer token when `CONTROLLER_API_TOKEN` is set:

```bash
curl -X POST http://localhost:8080/models/load \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.1-8b"}'
```

Aliases (`llama-3.1-8b`, `default`) resolve via the `model_aliases` SQLite table.

**Option C — reset persisted state**

```bash
./scripts/compose.sh down
docker volume rm inferedge_controller_data
./scripts/compose.sh up -d
```

## Inference via LiteLLM

Use **port 80** (Traefik → LiteLLM), not the controller on port 8080.

Once state is `READY`:

```bash
curl http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Architecture

```
Customer → Traefik (:80) → LiteLLM → inferedge-vllm:8000 (dynamic)
                              ↑
                         Controller (:8080)
                              ↓
                    SQLite + pluggable ServingBackend (litellm_vllm default)
                              ↓
                    vLLM container (Docker labels)
```

### Layering Rules

| Layer | Writes | Reads |
|---|---|---|
| API (`main.py`) | `intent_log` only | SQLite cache |
| Reconciler | `desired_state`, `appliance_state`, `deployments`, `reconcile_log` | `artifacts.py` + `ServingBackend` |
| `ServingBackend` | Runtime deployment (Docker or Ray Serve) | — |

### vLLM Label Identity

Containers are identified by labels, never names:

- `inferedge.config_hash` — hash of model + context + gpu_util
- `inferedge.generation` — monotonic counter per new container
- `inferedge.gpu_ids` — comma-separated GPU UUIDs

On unexpected exit, the last ~200 log lines and exit code are stored in `deployments`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CONTROLLER_API_TOKEN` | (empty) | Bearer token for `/models/load`; empty disables auth |
| `CONTROLLER_SHM_PCT` | 40 | **Source of truth** for controller `/dev/shm` size (% of host RAM; Ray needs >30%). `./scripts/compose.sh` writes derived `CONTROLLER_SHM_SIZE_GB` into `.env` before compose runs. |
| `VLLM_CONTAINER_STARTUP_TIMEOUT_SEC` | 120 | Container running-state timeout |
| `VLLM_PROBE_TIMEOUT_SEC` | 600 | `/health` + `/v1/models` probe timeout |
| `RECONCILE_INTERVAL_SEC` | 5 | Reconciler loop interval |
| `DOWNLOAD_CONNECTIONS` | 16 | Parallel HTTP connections per shard (aria2) |

Large model shards use **aria2** multi-connection download. Set `HF_TOKEN` for higher Hugging Face rate limits. At ~50 KB/s single-stream, a 2 GB shard takes ~11 hours; 16 connections typically reach much higher throughput.

## Schema Migrations

Schema version is tracked in `schema_meta`. To add a migration:

1. Increment `SCHEMA_VERSION` in `controller/state.py`
2. Add a `_migration_vN` function and register it in `MIGRATIONS`

## Acceptance Criteria

1. Cold boot reaches `READY` with working LiteLLM endpoint
2. `/status` responsive from t=0 with updating `last_reconcile_ts`
3. Reboot restores previous model from SQLite desired state
4. Docker daemon restart handled by reconciler recreation
5. Conflicting `/models/load` calls resolved via intent log ordering
6. Cache/disk errors → `DEGRADED` with actionable `last_error`
7. CPU-only → `DEGRADED` (not `FAILED`)
8. Exactly one vLLM container with correct labels
9. End-to-end inference via LiteLLM once `READY`
10. `reconcile_log` grows only on real changes

## Troubleshooting

| State | Likely cause | Action |
|---|---|---|
| `DEGRADED` + GPU message | No NVIDIA GPU / driver | Install drivers + nvidia-container-toolkit |
| `DEGRADED` + HF auth (401) | No/invalid token | Set `HF_TOKEN` in `.env` |
| `DEGRADED` + HF access denied (403) | Token set, license not accepted | Log into HF as token owner → open model page → click **Agree and access repository** |
| `DEGRADED` + disk full | Cache volume full | Free space under `LOCAL_MODEL_CACHE` |
| `FAILED` + Docker error | Daemon / GPU passthrough | Check `docker compose logs controller` and `deployments.log_snippet` in `/status` |
| `RECONCILING` (long) | First model download | Check `curl localhost:8080/status` for `download_bytes`; files appear under `MODEL_CACHE_HOST` on host |
| Empty `MODEL_CACHE_HOST` during download | Cache path bug (old builds) | Rebuild controller; ensure compose sets `LOCAL_MODEL_CACHE=/models_cache` inside container |
| Ray `/dev/shm` performance warning | `shm_size` too small | Raise `CONTROLLER_SHM_PCT` (≥35) in `.env`, then `./scripts/compose.sh up -d --force-recreate controller` |
| Download interrupted, exit code 137 | Container killed (SIGKILL) | Usually `docker compose --force-recreate` or OOM kill — download resumes from partial cache; avoid recreating controller mid-download |
| `FAILED` + vLLM name conflict (409) | Stale `inferedge-vllm-gen*` in `created` state | Controller auto-heals on retry; or `docker rm -f inferedge-vllm-gen1` then wait for reconcile |
| `DEGRADED` + CUDA 804 / driver mismatch | vLLM image CUDA newer than host driver | Upgrade NVIDIA driver (≥550 for CUDA 12.4 images) or pin older `VLLM_IMAGE`; retry via `POST /models/load` |
| `POST /v1/chat/completions` 404 on :8080 | Wrong port | Use `http://localhost/v1/...` (port 80), not the controller API |