#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=artifacts/operations
exec .venv/bin/python -u artifacts/operations/grouped_resume_20261015.py --root /ceph/aasteine/decision-em-20261013 --admission /ceph/aasteine/decision-em-20261013/configs/grouped_confirmation_admission_20261015.json
