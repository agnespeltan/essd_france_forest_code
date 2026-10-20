#!/usr/bin/env bash
# Run the full age-map pipeline on this machine.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-./config.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG — copy config.example.yaml and edit paths:"
  echo "  cp config.example.yaml config.yaml"
  exit 1
fi

# Reuse existing GRECO weights by default (set SKIP_WEIGHTS=0 to rebuild)
SKIP_WEIGHTS="${SKIP_WEIGHTS:-1}"
WITH_FIGURES="${WITH_FIGURES:-0}"

ARGS=(--config "$CONFIG")
if [[ "$SKIP_WEIGHTS" == "1" ]]; then
  ARGS+=(--skip-weights)
fi
if [[ "$WITH_FIGURES" == "1" ]]; then
  ARGS+=(--with-figures)
fi
# Extra flags passed through, e.g. ./run_local.sh --force --max-blocks 20
ARGS+=("$@")

echo "Local run: python3 05_run_all.py ${ARGS[*]}"
exec python3 05_run_all.py "${ARGS[@]}"
