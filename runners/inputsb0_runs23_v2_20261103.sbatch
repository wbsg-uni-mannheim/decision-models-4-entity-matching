#!/bin/bash
# Input variants with B0, runs 2/3 of F1-F4, R2-R4, R6. Usage: sbatch inputsb0_runs23_20261103.sbatch --run 2|3 [--resume]
# Needs: frozen manifest per run (--action freeze), Aaron's approval file, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>.
#SBATCH --job-name=em-inputsb0-runs23-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/path/to/decision-em/logs/inputsb0-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_inputsb0_runs23_v2_20261103.py --root /path/to/decision-em --action execute "$@"
