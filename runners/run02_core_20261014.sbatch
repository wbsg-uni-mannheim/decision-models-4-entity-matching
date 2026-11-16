#!/bin/bash
set -euo pipefail
cd /path/to/decision-em
exec .venv/bin/python -u artifacts/operations/repeated_runs_parallel.py --root /path/to/decision-em --run-id 2 --stage core --admission /path/to/decision-em/configs/run02_core_admission_20261014.json --parallel-policy /path/to/decision-em/configs/parallel_execution_20261014.json
