#!/bin/bash
# Inputs study (F1-F4 serialization, R1-R6 robustness), one run on the full core test sets.
# Needs: frozen prompt selection, frozen manifest, Aaron's approval file, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>. Resume: add --resume to the command below.
#SBATCH --job-name=em-inputs-run01
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/inputs-run01-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_inputs_20261019.py --root /ceph/aasteine/decision-em-20261013 --action execute "$@"
