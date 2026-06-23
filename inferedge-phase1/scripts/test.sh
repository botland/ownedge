#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker build -q -t inferedge-controller-test controller

docker run --rm \
  -v "$ROOT/inferedge-phase1/controller:/app" \
  -v "$ROOT/inferedge-phase1/coverage-html:/coverage-html" \
  -w /app \
  inferedge-controller-test \
  sh -c "pip install -q pytest pytest-asyncio pytest-cov pytest-mock && python -m pytest tests --cov=. --cov-report=term-missing --cov-report=html:/coverage-html -v $*"