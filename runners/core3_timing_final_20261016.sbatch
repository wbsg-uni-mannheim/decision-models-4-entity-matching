#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=artifacts/operations
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -u artifacts/operations/complete_core3_timing_20261016.py --root /path/to/decision-em --admission /path/to/decision-em/configs/core3_timing_admission_20261016.json
