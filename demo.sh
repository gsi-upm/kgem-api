#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

wait_for() {
  local service="$1"
  local url="$2"
  curl --retry 60 --retry-all-errors --retry-delay 1 --fail --silent "$url" >/dev/null || {
    echo "No se pudo iniciar $service ($url)." >&2
    return 1
  }
}

docker compose build

if [[ ! -f kgem_api/models/nations_transe/trained_model.pkl ]]; then
  docker compose run --rm --no-deps \
    --volume "$PWD/kgem_api/scripts:/app/scripts:ro" \
    kgem-api uv run python scripts/model_training.py
fi

docker compose up -d
wait_for "KGEM-API" "http://localhost:8000/"
wait_for "el recomendador" "http://localhost:8002/docs"
wait_for "la interfaz" "http://localhost:8501/_stcore/health"

echo "Demo lista en http://localhost:8501"
