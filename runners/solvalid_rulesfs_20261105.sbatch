#!/bin/bash
# PPLX v1.2 RULESFS10_2000 on the validation split of the Sol 2,000 set (3,200 pairs), one run (cascade threshold).
# Needs: frozen manifest (--action freeze), Aaron's approval file, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>. Resume: add --resume to the command below.
#SBATCH --job-name=em-solvalid-rfs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/path/to/decision-em/logs/solvalid-rfs-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_solvalid_rulesfs_20261105.py --root /path/to/decision-em --action execute "$@"
