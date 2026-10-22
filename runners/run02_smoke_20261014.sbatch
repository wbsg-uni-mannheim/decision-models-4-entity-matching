#!/bin/bash
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
exec .venv/bin/python -u artifacts/operations/repeated_runs_parallel.py --root /ceph/aasteine/decision-em-20261013 --run-id 2 --stage smoke --admission /ceph/aasteine/decision-em-20261013/configs/run02_smoke_admission_20261014.json --parallel-policy /ceph/aasteine/decision-em-20261013/configs/parallel_execution_20261014.json
