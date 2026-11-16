#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
exec .venv/bin/python -u artifacts/operations/authorized_recovery_200_reviewed.py --root /path/to/decision-em --stage core --run-id 3 --admission /path/to/decision-em/configs/core_run03_budget200_reviewed_admission_20261015.json
