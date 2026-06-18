#!/usr/bin/env bash
set -euo pipefail

PORT="${CONTROLLER_PORT:-8080}"
STATE=$(curl -sf "http://localhost:${PORT}/status" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")

if [[ "$STATE" == "READY" || "$STATE" == "DEGRADED" ]]; then
  exit 0
fi
exit 1