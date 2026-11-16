#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations
exec .venv/bin/python -u artifacts/operations/resume_fresh_core_run02_third_drained.py --root /path/to/decision-em --admission /path/to/decision-em/configs/core_run02_third_admission_20261015.json
