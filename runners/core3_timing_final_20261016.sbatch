#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=artifacts/operations
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -u artifacts/operations/complete_core3_timing_20261016.py --root /ceph/aasteine/decision-em-20261013 --admission /ceph/aasteine/decision-em-20261013/configs/core3_timing_admission_20261016.json
