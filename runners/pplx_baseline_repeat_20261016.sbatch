#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=artifacts/operations
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/run_pplx_baseline_repeat_20261016.py --root /ceph/aasteine/decision-em-20261013 --admission /ceph/aasteine/decision-em-20261013/configs/pplx_repeat_admission_20261016.json
