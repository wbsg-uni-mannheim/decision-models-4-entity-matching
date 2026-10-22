#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations
exec .venv/bin/python -u artifacts/operations/resume_variants_third_drained.py --root /ceph/aasteine/decision-em-20261013 --admission /ceph/aasteine/decision-em-20261013/configs/variants_third_admission_20261015.json
