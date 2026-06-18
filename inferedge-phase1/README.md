# InferEdge Phase 1

Single-node inference appliance with a Controller reconciler, label-managed vLLM, LiteLLM gateway, and Traefik routing.

## Prerequisites

- Ubuntu 24.04 (or similar Linux)
- NVIDIA GPU with ≥8 GB VRAM (≥16 GB recommended)
- Docker Compose v2
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Optional: Hugging Face token for gated models (`HF_TOKEN`)

## Quick Start

```bash
cp .env.example .env
# Edit .env: set CONTROLLER_API_TOKEN and HF_TOKEN if needed

docker compose up -d --build
docker compose logs -f controller
```

## Check Status

```bash
curl http://localhost:8080/status | jq
```

`/status` is available immediately on boot (state `BOOT` → `RECONCILING` → `READY`).

## Load a Model

Protected endpoint — requires Bearer token when `CONTROLLER_API_TOKEN` is set:

```bash
curl -X POST http://localhost:8080/models/load \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.1-8b"}'
```

Aliases (`llama-3.1-8b`, `default`) resolve via the `model_aliases` SQLite table.

## Inference via LiteLLM

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
                    SQLite + Ray (compute adapter)
                              ↓
                    vLLM container (Docker labels)
```

### Layering Rules

| Layer | Writes | Reads |
|---|---|---|
| API (`main.py`) | `intent_log` only | SQLite cache |
| Reconciler | `desired_state`, `appliance_state`, `deployments`, `reconcile_log` | Docker via `models.py` |
| `models.py` | Docker containers | — |

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
| `VLLM_CONTAINER_STARTUP_TIMEOUT_SEC` | 120 | Container running-state timeout |
| `VLLM_PROBE_TIMEOUT_SEC` | 600 | `/health` + `/v1/models` probe timeout |
| `RECONCILE_INTERVAL_SEC` | 5 | Reconciler loop interval |

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
| `DEGRADED` + HF auth | Gated model | Set `HF_TOKEN` in `.env` |
| `DEGRADED` + disk full | Cache volume full | Free space under `LOCAL_MODEL_CACHE` |
| `FAILED` + Docker error | Daemon / GPU passthrough | Check `docker compose logs controller` and `deployments.log_snippet` in `/status` |
| `RECONCILING` (long) | First model download | Wait; monitor `docker compose logs controller` |