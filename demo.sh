#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

docker compose build

if [[ ! -f kgem_api/models/nations_transe/trained_model.pkl ]]; then
  docker compose run --rm --no-deps \
    --volume "$PWD/kgem_api/scripts:/app/scripts:ro" \
    kgem-api uv run python scripts/model_training.py
fi

docker compose up -d
curl --retry 60 --retry-connrefused --retry-delay 1 --fail --silent --show-error \
  http://localhost:8002/docs >/dev/null

curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"news1_entities":["usa","uk"],"news2_entities":["usa","netherlands"],"mode":"regression","model_name":"random_forest","graph":"nations","embedding_model":"transe"}' \
  http://localhost:8002/recommend/
echo
