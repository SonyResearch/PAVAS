#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -B tools/smoke_test_imports.py
python3 -B tools/smoke_test_paths.py
python3 -B tools/smoke_test_ppe_inputs.py
python3 -B tools/smoke_test_ppe_wrappers.py
python3 -B tools/smoke_test_ppe_pipeline.py
python3 -B tools/smoke_test_clip_loader.py
python3 -B tools/smoke_test_infer.py
