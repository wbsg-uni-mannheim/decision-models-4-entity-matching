#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
exec .venv/bin/python -u artifacts/operations/authorized_recovery_200.py --root /ceph/aasteine/decision-em-20261013 --stage variants --run-id 1 --admission /ceph/aasteine/decision-em-20261013/configs/variants_budget200_admission_20261015.json
