#!/bin/bash
# Prompt selection on the full validation sets: 10 prompts x 7 models x 8 benchmarks, one run.
# Needs: frozen manifest (--action freeze), Aaron's approval file, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>. Resume: add --resume to the command below.
#SBATCH --job-name=em-promptval-run01
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/path/to/decision-em/logs/promptval-run01-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_promptval_20261027.py --root /path/to/decision-em --action execute "$@"
