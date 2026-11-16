#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
exec .venv/bin/python -u artifacts/operations/authorized_recovery_200_reviewed.py --root /path/to/decision-em --stage core --run-id 2 --admission /path/to/decision-em/configs/core_run02_budget200_reviewed_admission_20261015.json
