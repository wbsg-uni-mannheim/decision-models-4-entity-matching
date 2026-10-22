#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=artifacts/operations
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/same_condition_remainder_20261017.py --root /ceph/aasteine/decision-em-20261013 launch
