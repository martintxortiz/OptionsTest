#!/bin/sh
set -eu

CPU_WORKERS="${CPU_WORKERS:-$(nproc)}"
HEAVY_PRESET="${HEAVY_PRESET:-cpu_max}"
MONTE_CARLO_ITERATIONS="${MONTE_CARLO_ITERATIONS:-5000}"

exec python run_research_pipeline.py \
  --heavy-preset "${HEAVY_PRESET}" \
  --cpu-workers "${CPU_WORKERS}" \
  --mc-iterations "${MONTE_CARLO_ITERATIONS}" \
  "$@"
