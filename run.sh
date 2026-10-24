#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop, then run this script again." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required." >&2
  exit 1
fi

mkdir -p data backend/app/ml/models
docker compose build backend

if [[ ! -f backend/app/ml/models/training_metrics.json ]]; then
  echo "Training all 12 endpoint models. The first run downloads Tox21 and may take a few minutes."
  docker compose run --rm backend python -m app.ml.train
fi

docker compose up --build -d
echo "Dashboard: http://localhost:${WEB_PORT:-8080}"
echo "API docs:  http://localhost:${API_PORT:-8000}/docs"