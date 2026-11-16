#!/bin/bash
# B0 on the Sol-labelled 200 and 2,000 sets: 6 decision models x 8 benchmarks, one run (threshold tuning).
# Needs: frozen manifest (--action freeze), Aaron's approval file, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>. Resume: add --resume to the command below.
#SBATCH --job-name=em-solset-b0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/path/to/decision-em/logs/solset-b0-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_solset_b0_20261105.py --root /path/to/decision-em --action execute "$@"
