#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=artifacts/operations
exec .venv/bin/python -u artifacts/operations/fast_resume_20261015.py --root /path/to/decision-em --admission /path/to/decision-em/configs/fast_core2_directory_admission_20261015.json
