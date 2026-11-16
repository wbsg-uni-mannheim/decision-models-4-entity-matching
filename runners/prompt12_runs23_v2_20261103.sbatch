#!/bin/bash
#SBATCH --job-name=prompt12-runs23-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/path/to/decision-em/logs/prompt12-runs23-%j.log
# usage: sbatch artifacts/operations/prompt12_runs23_20261103.sbatch --run 2|3 [--resume] [--models ...]
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_prompt12_runs23_v2_20261103.py --root /path/to/decision-em --action execute "$@"
