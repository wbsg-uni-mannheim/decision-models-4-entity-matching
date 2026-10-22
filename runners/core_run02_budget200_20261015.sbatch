#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
exec .venv/bin/python -u artifacts/operations/authorized_recovery_200.py --root /ceph/aasteine/decision-em-20261013 --stage core --run-id 2 --admission /ceph/aasteine/decision-em-20261013/configs/core_run02_budget200_admission_20261015.json
