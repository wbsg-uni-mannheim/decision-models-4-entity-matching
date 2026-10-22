#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
exec .venv/bin/python -u artifacts/operations/authorized_recovery_200_reviewed.py --root /ceph/aasteine/decision-em-20261013 --stage schema --run-id 1 --admission /ceph/aasteine/decision-em-20261013/configs/schema_budget200_reviewed_admission_20261015.json
