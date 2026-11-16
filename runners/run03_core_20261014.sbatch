#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
exec .venv/bin/python -u artifacts/operations/repeated_runs_parallel.py --root /path/to/decision-em --run-id 3 --stage core --admission /path/to/decision-em/configs/run03_core_admission_20261014.json --parallel-policy /path/to/decision-em/configs/parallel_execution_20261014.json
