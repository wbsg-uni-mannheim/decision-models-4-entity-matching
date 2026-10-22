#!/bin/bash
# Escalation of uncertain test pairs to the three-model ensemble, one run.
# Full set: no extra argument. 500-pair subsample: --sample. Resume: add --resume.
# Needs: frozen manifest (--action freeze [--sample]), Aaron's approval file, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>.
#SBATCH --job-name=em-escalation-run01
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/escalation-run01-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_escalation_20261020.py --root /ceph/aasteine/decision-em-20261013 --action execute "$@"
