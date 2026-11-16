#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=artifacts/operations
exec .venv/bin/python -u artifacts/operations/resume_grouped_core3_third_20261016.py --root /path/to/decision-em --admission /path/to/decision-em/configs/grouped_core3_third_admission_20261016.json
