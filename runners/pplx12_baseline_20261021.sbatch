#!/bin/bash
# PPLX v1.2, B0, full test sets, collections 2 and 3 (sequential). Resume: add --resume.
#SBATCH --job-name=em-pplx12-b0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/path/to/decision-em/logs/pplx12-b0-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
for r in 2 3; do
  .venv/bin/python -u artifacts/operations/run_pplx12_baseline_20261021.py --root /path/to/decision-em --action execute --run-id $r "$@"
done
