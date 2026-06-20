#!/usr/bin/env bash
# Compute CONTROLLER_SHM_SIZE_GB from CONTROLLER_SHM_PCT + host RAM and write to .env.
# CONTROLLER_SHM_PCT is the single source of truth; CONTROLLER_SHM_SIZE_GB is derived.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
EXAMPLE_FILE="${ROOT}/.env.example"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$EXAMPLE_FILE" ]]; then
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    echo "Created ${ENV_FILE} from .env.example" >&2
  else
    echo "error: ${ENV_FILE} not found" >&2
    exit 1
  fi
fi

pct="${CONTROLLER_SHM_PCT:-}"
if [[ -z "$pct" ]]; then
  val="$(grep -E '^CONTROLLER_SHM_PCT=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
  [[ -n "$val" ]] && pct="$val"
fi
pct="${pct:-40}"

total_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
shm_gb="$(( (total_kb * 1024 * pct / 100 + 1073741824 - 1) / 1073741824 ))"
(( shm_gb < 1 )) && shm_gb=1

current_gb="$(grep -E '^CONTROLLER_SHM_SIZE_GB=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
if [[ "$current_gb" == "$shm_gb" ]]; then
  if [[ "${RESOLVE_SHM_VERBOSE:-}" == "1" ]]; then
    total_gb="$(awk -v kb="$total_kb" 'BEGIN { printf "%.1f", kb / 1024 / 1024 }')"
    echo "CONTROLLER_SHM_SIZE_GB=${shm_gb} unchanged (${pct}% of ${total_gb} GB RAM)" >&2
  fi
  exit 0
fi

if grep -qE '^CONTROLLER_SHM_SIZE_GB=' "$ENV_FILE"; then
  sed -i "s|^CONTROLLER_SHM_SIZE_GB=.*|CONTROLLER_SHM_SIZE_GB=${shm_gb}|" "$ENV_FILE"
else
  if grep -qE '^CONTROLLER_SHM_PCT=' "$ENV_FILE"; then
    sed -i "/^CONTROLLER_SHM_PCT=/a CONTROLLER_SHM_SIZE_GB=${shm_gb}" "$ENV_FILE"
  else
    printf '\nCONTROLLER_SHM_SIZE_GB=%s\n' "$shm_gb" >> "$ENV_FILE"
  fi
fi

if [[ "${RESOLVE_SHM_VERBOSE:-}" == "1" ]]; then
  total_gb="$(awk -v kb="$total_kb" 'BEGIN { printf "%.1f", kb / 1024 / 1024 }')"
  echo "Updated .env: CONTROLLER_SHM_PCT=${pct}% of ${total_gb} GB RAM -> CONTROLLER_SHM_SIZE_GB=${shm_gb}" >&2
fi