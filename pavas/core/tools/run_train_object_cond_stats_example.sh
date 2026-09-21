#!/usr/bin/env bash
# Example wrapper to compute train split object condition stats.
# Adjust the ROOT paths below to your actual train mass / velocity roots.
# Usage:
#   bash core/tools/run_train_object_cond_stats_example.sh

set -euo pipefail

MASS_ROOT="/path/to/ppe/mass"
VELOCITY_ROOT="/path/to/ppe/velocity"
OUT_JSON="stats/object_cond_train.json"

mkdir -p "$(dirname "$OUT_JSON")"

python -m core.tools.compute_object_condition_stats \
  --mass-root "$MASS_ROOT" \
  --velocity-root "$VELOCITY_ROOT" \
  --out "$OUT_JSON" \
  --verbose

echo "\n[done] Wrote stats to $OUT_JSON"
