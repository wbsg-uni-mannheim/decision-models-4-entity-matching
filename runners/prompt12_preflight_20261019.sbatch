#!/bin/bash
# Read-only preflight for the prompt12 run: no API calls, no ledger writes.
#SBATCH --job-name=em-prompt12-preflight
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/prompt12-preflight-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_prompt12_20261019.py --root /ceph/aasteine/decision-em-20261013 --action preflight
